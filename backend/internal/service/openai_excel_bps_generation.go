package service

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
	"github.com/Wei-Shaw/sub2api/internal/pkg/openai"
	"github.com/Wei-Shaw/sub2api/internal/service/basispoints"
	"github.com/Wei-Shaw/sub2api/internal/util/transportdiag"
	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
	"go.uber.org/zap"
)

type excelBPSImageSlotKey struct{}

// WithExcelBPSImageSlotAcquirer defers the existing handler image limit until
// the model actually selects the server-owned tool. Ordinary chat holds no slot.
func WithExcelBPSImageSlotAcquirer(ctx context.Context, acquire func(context.Context) (func(), bool)) context.Context {
	return context.WithValue(ctx, excelBPSImageSlotKey{}, acquire)
}

type excelBPSImageGenerationState struct {
	mu     sync.Mutex
	result *OpenAIForwardResult
}

func (s *OpenAIGatewayService) excelBPSImageGenerator(ctx context.Context, c *gin.Context, account *Account, model string, state *excelBPSImageGenerationState) basispoints.ImageGenerator {
	apiKey := getAPIKeyFromContext(c)
	if apiKey == nil || !GroupAllowsImageGeneration(apiKey.Group) ||
		isOpenAIResponsesCompactPath(c) ||
		isCodexSparkModel(account.GetMappedModel(model)) ||
		account.CodexImageGenerationExplicitToolPolicy() == codexImageGenerationExplicitToolPolicyStrip ||
		!s.isCodexImageGenerationBridgeEnabled(ctx, account, apiKey) ||
		!(openai.IsCodexOfficialClientByHeaders(c.GetHeader("User-Agent"), c.GetHeader("originator")) || (s.cfg != nil && s.cfg.Gateway.ForceCodexCLI)) {
		return nil
	}
	acquire, _ := ctx.Value(excelBPSImageSlotKey{}).(func(context.Context) (func(), bool))
	// Entry points without the handler's limiter must not bypass an enabled limit.
	if acquire == nil && s.cfg != nil && s.cfg.Gateway.ImageConcurrency.Enabled {
		return nil
	}
	// BPS lowers client tools to its own protocol, including Responses Lite
	// clients. The server-owned tool can therefore be offered on this path.
	// Its child request is a fresh, non-Lite native hosted-image request; the
	// native Lite passthrough restrictions elsewhere remain authoritative.
	snapshot := c.Copy()
	return func(request basispoints.ImageGenerationRequest) (basispoints.ImageGenerationResult, error) {
		started := time.Now()
		imageCtx, cancel := context.WithTimeout(ctx, 4*time.Minute)
		defer cancel()
		if err := imageCtx.Err(); err != nil {
			return basispoints.ImageGenerationResult{}, err
		}
		if acquire != nil {
			release, ok := acquire(imageCtx)
			if !ok {
				return basispoints.ImageGenerationResult{}, errors.New("image generation concurrency limit exceeded")
			}
			if release != nil {
				defer release()
			}
		}
		body, err := json.Marshal(map[string]any{
			"model": model, "stream": true, "store": false,
			"instructions": "Generate exactly one image matching the user's prompt using the image_generation tool.",
			"input":        request.Prompt,
			"tools":        []any{map[string]any{"type": "image_generation", "size": request.Size, "quality": request.Quality, "output_format": "png"}},
			"tool_choice":  map[string]any{"type": "image_generation"},
		})
		if err != nil {
			return basispoints.ImageGenerationResult{}, err
		}
		w := &excelBPSImageRecorder{ResponseRecorder: httptest.NewRecorder(), cancel: cancel}
		child, _ := gin.CreateTestContext(w)
		child.Keys = snapshot.Keys
		// This is a new HTTP image request, not a replay of the outer text
		// request. Reinitialize its permission and billing classification.
		SetOpenAIClientTransport(child, OpenAIClientTransportHTTP)
		SetOpenAIImageIntentHint(child, true)
		child.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body)).WithContext(imageCtx)
		child.Request.Header.Set("Content-Type", "application/json")
		child.Request.Header.Set("User-Agent", snapshot.GetHeader("User-Agent"))
		child.Request.Header.Set("originator", snapshot.GetHeader("originator"))
		// The explicit hosted declaration routes this one call natively. No BPS
		// flag is changed, and Forward retains native auth, proxy and account policy.
		forward, forwardErr := s.Forward(imageCtx, child, account, body)
		state.mu.Lock()
		state.result = forward
		state.mu.Unlock()
		failure := func(reason, terminal string, response map[string]any, decodeErr error) {
			logExcelBPSImageFailure(imageCtx, account.ID, reason, w, forward, forwardErr, decodeErr, terminal, response, started)
		}
		var response map[string]any
		responseBytes := w.Body.Bytes()
		terminalEvent := ""
		if !gjson.ValidBytes(responseBytes) {
			var terminal []byte
			var ok bool
			terminalEvent, terminal, ok = extractOpenAISSETerminalEvent(w.Body.String())
			if !ok {
				failure("missing_terminal", terminalEvent, nil, nil)
				return basispoints.ImageGenerationResult{}, errors.New("native image stream has no terminal response")
			}
			responseBytes = []byte(gjson.GetBytes(terminal, "response").Raw)
		}
		decoder := json.NewDecoder(bytes.NewReader(responseBytes))
		decoder.UseNumber()
		decodeErr := decoder.Decode(&response)
		generated := basispoints.ImageGenerationResult{}
		if response != nil {
			generated.Usage, _ = response["usage"].(map[string]any)
			generated.ToolUsage, _ = response["tool_usage"].(map[string]any)
		}
		if forwardErr != nil || w.overflow || decodeErr != nil || w.Code != http.StatusOK || response["status"] != "completed" {
			reason := "response_status"
			switch {
			case w.overflow:
				reason = "response_size_limit"
			case w.Code != http.StatusOK:
				reason = "http_status"
			case forwardErr != nil:
				reason = "forward_error"
			case decodeErr != nil:
				reason = "decode_error"
			}
			failure(reason, terminalEvent, response, decodeErr)
			return generated, errors.New("native image generation did not complete")
		}
		output, _ := response["output"].([]any)
		for _, raw := range output {
			item, _ := raw.(map[string]any)
			if item["type"] == "image_generation_call" && item["status"] == "completed" {
				if generated.Item != nil {
					failure("multiple_images", terminalEvent, response, nil)
					return generated, errors.New("native generation returned multiple images")
				}
				generated.Item = item
			}
		}
		if generated.Item == nil {
			failure("missing_image_item", terminalEvent, response, nil)
			return generated, errors.New("native generation returned no completed image")
		}
		if image, _ := generated.Item["result"].(string); image == "" {
			failure("empty_image_result", terminalEvent, response, nil)
			return generated, errors.New("native generation returned empty image")
		}
		return generated, nil
	}
}

// This intentionally logs only bounded metadata. Upstream messages and image
// bytes may include prompts, credentials, or generated media.
func logExcelBPSImageFailure(ctx context.Context, accountID int64, reason string, recorder *excelBPSImageRecorder, forward *OpenAIForwardResult, forwardErr, decodeErr error, terminal string, response map[string]any, started time.Time) {
	forwardKind := ""
	if forwardErr != nil {
		forwardKind = transportdiag.Classify(forwardErr)
	}
	forwardTerminal := ""
	if forward != nil {
		forwardTerminal = forward.UpstreamTerminalEvent
	}
	output, _ := response["output"].([]any)
	errorObject, _ := response["error"].(map[string]any)
	logger.FromContext(ctx).Warn("excel_bps.native_image_child_failed",
		zap.Int64("account_id", accountID),
		zap.String("reason", reason),
		zap.Int("http_status", recorder.Code),
		zap.String("terminal_event", excelBPSImageDiagnosticEvent(terminal)),
		zap.String("forward_terminal_event", excelBPSImageDiagnosticEvent(forwardTerminal)),
		zap.String("response_status", excelBPSImageDiagnosticStatus(response["status"])),
		zap.String("upstream_error_code", excelBPSImageDiagnosticCode(errorObject["code"])),
		zap.String("upstream_error_type", excelBPSImageDiagnosticType(errorObject["type"])),
		zap.String("upstream_error_param", excelBPSImageDiagnosticParam(errorObject["param"])),
		zap.String("forward_error_kind", forwardKind),
		zap.Bool("decode_error", decodeErr != nil),
		zap.Bool("response_size_limit", recorder.overflow),
		zap.Int("response_bytes", recorder.Body.Len()),
		zap.Int("output_items", len(output)),
		zap.Int64("duration_ms", time.Since(started).Milliseconds()),
	)
}

func excelBPSImageDiagnosticEvent(event string) string {
	switch event {
	case "response.completed", "response.failed", "response.incomplete", "response.cancelled", "response.canceled", "response.done", "error":
		return event
	case "":
		return ""
	default:
		return "other"
	}
}

func excelBPSImageDiagnosticStatus(value any) string {
	switch value {
	case "completed", "failed", "incomplete", "cancelled", "canceled":
		return value.(string)
	case nil:
		return ""
	default:
		return "other"
	}
}

func excelBPSImageDiagnosticCode(value any) string {
	switch value {
	case "server_error", "invalid_request_error", "rate_limit_exceeded", "insufficient_quota", "model_not_found", "unsupported_model", "image_generation_failed", "content_policy_violation", "safety_violation", "timeout", "internal_error", "upstream_error", "overloaded", "capacity_error":
		return value.(string)
	case nil, "":
		return ""
	default:
		return "other"
	}
}

func excelBPSImageDiagnosticType(value any) string {
	switch value {
	case "invalid_request_error", "authentication_error", "permission_error", "rate_limit_error", "server_error", "api_error":
		return value.(string)
	case nil, "":
		return ""
	default:
		return "other"
	}
}

func excelBPSImageDiagnosticParam(value any) string {
	switch value {
	case "model", "tools", "tools[0].model", "tools[0].size", "tools[0].quality", "tools[0].output_format", "tool_choice", "input", "instructions", "stream", "store":
		return value.(string)
	case nil, "":
		return ""
	default:
		return "other"
	}
}

func (s *excelBPSImageGenerationState) apply(result *OpenAIForwardResult) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.result == nil || result == nil {
		return
	}
	native := s.result
	result.ImageCount, result.ImageSize = native.ImageCount, native.ImageSize
	result.ImageInputSize, result.ImageOutputSize = native.ImageInputSize, native.ImageOutputSize
	result.ImageOutputSizes, result.ImageSizeSource = native.ImageOutputSizes, native.ImageSizeSource
	result.ImageSizeBreakdown = native.ImageSizeBreakdown
	if native.ImageCount > 0 {
		result.BillingModel = native.BillingModel
	}
}

type excelBPSImageRecorder struct {
	*httptest.ResponseRecorder
	cancel   context.CancelFunc
	overflow bool
}

func (w *excelBPSImageRecorder) Write(data []byte) (int, error) {
	// Native SSE can contain the image in both output_item.done and completed.
	if w.overflow || w.Body.Len()+len(data) > 32<<20 {
		w.overflow = true
		w.cancel()
		return 0, io.ErrShortWrite
	}
	return w.ResponseRecorder.Write(data)
}
func (w *excelBPSImageRecorder) WriteString(data string) (int, error) { return w.Write([]byte(data)) }
