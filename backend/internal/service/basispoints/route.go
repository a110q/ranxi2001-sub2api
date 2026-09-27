package basispoints

import (
	"strings"

	"github.com/tidwall/gjson"
)

// NativeFallbackReason reports capabilities that must stay on the native Codex
// channel because Basispoints cannot execute them. Empty means the request can
// use the BPS bridge.
func NativeFallbackReason(body []byte) string {
	if !gjson.ValidBytes(body) {
		return ""
	}
	var inspectTools func(gjson.Result) string
	inspectTools = func(tools gjson.Result) string {
		if !tools.IsArray() {
			return ""
		}
		fallback := ""
		tools.ForEach(func(_, tool gjson.Result) bool {
			kind := strings.ToLower(strings.TrimSpace(tool.Get("type").String()))
			switch kind {
			case "namespace":
				fallback = inspectTools(tool.Get("tools"))
				if fallback != "" {
					return false
				}
			case "image_generation":
				fallback = "image_generation"
				return false
			case "web_search", "web_search_preview", "web_search_preview_2025_03_11", "web_search_2025_08_26":
				if tool.Get("external_web_access").Bool() || tool.Get("search_context_size").String() == "high" {
					fallback = "web_search"
					return false
				}
			}
			return true
		})
		return fallback
	}
	if reason := inspectTools(gjson.GetBytes(body, "tools")); reason != "" {
		return reason
	}
	// Responses Lite may carry declarations in input.additional_tools. Prepare
	// collects those declarations too, so route hosted tools before it filters
	// capabilities that BPS cannot execute. Passive client image_gen functions
	// remain ordinary BPS tools.
	if input := gjson.GetBytes(body, "input"); input.IsArray() {
		fallback := ""
		input.ForEach(func(_, item gjson.Result) bool {
			if item.Get("type").String() == "additional_tools" {
				fallback = inspectTools(item.Get("tools"))
			}
			return fallback == ""
		})
		if fallback != "" {
			return fallback
		}
	}
	choice := gjson.GetBytes(body, "tool_choice")
	if choice.Exists() && choice.Type == gjson.JSON {
		if strings.EqualFold(choice.Get("type").String(), "image_generation") {
			return "image_generation"
		}
		if reason := inspectTools(choice.Get("tools")); reason != "" {
			return reason
		}
		name := choice.Get("name").String()
		if strings.Contains(strings.ToLower(name), "web_search") || strings.Contains(strings.ToLower(name), "image_generation") {
			return "tool_choice"
		}
	}
	// Inline data images are handled by Sub2API's local relay before Prepare;
	// leave them on BPS so the relay can rewrite them to signed HTTPS URLs.
	return ""
}
