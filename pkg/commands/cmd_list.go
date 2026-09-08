package commands

import (
	"context"
	"fmt"
	"strings"
)

func listCommand() Definition {
	return Definition{
		Name:        "list",
		Description: "List available options",
		SubCommands: []SubCommand{
			{
				Name:        "models",
				Description: "Configured models",
				Handler: func(_ context.Context, req Request, rt *Runtime) error {
					if rt == nil {
						return req.Reply(unavailableMsg)
					}
					if rt.ListModelPresets != nil {
						if presets := rt.ListModelPresets(); len(presets) > 0 {
							return req.Reply(formatModelPresets(presets))
						}
					}
					// No model_list, or a runtime that predates it: report the
					// single running model as before.
					if rt.GetModelInfo == nil {
						return req.Reply(unavailableMsg)
					}
					name, provider := rt.GetModelInfo()
					if provider == "" {
						provider = "configured default"
					}
					return req.Reply(fmt.Sprintf(
						"Configured Model: %s\nProvider: %s\n\nTo change models, update config.json",
						name, provider,
					))
				},
			},
			{
				Name:        "channels",
				Description: "Enabled channels",
				Handler: func(_ context.Context, req Request, rt *Runtime) error {
					if rt == nil || rt.GetEnabledChannels == nil {
						return req.Reply(unavailableMsg)
					}
					enabled := rt.GetEnabledChannels()
					if len(enabled) == 0 {
						return req.Reply("No channels enabled")
					}
					return req.Reply(fmt.Sprintf("Enabled Channels:\n- %s", strings.Join(enabled, "\n- ")))
				},
			},
			{
				Name:        "agents",
				Description: "Registered agents",
				Handler:     agentsHandler(),
			},
			{
				Name:        "skills",
				Description: "Installed skills",
				Handler: func(_ context.Context, req Request, rt *Runtime) error {
					if rt == nil || rt.ListSkillNames == nil {
						return req.Reply(unavailableMsg)
					}
					names := rt.ListSkillNames()
					if len(names) == 0 {
						return req.Reply("No installed skills")
					}
					return req.Reply(fmt.Sprintf(
						"Installed Skills:\n- %s\n\nUse /use <skill> <message> to force one for a single request, or /use <skill> to apply it to your next message.",
						strings.Join(names, "\n- "),
					))
				},
			},
			{
				Name:        "mcp",
				Description: "Configured MCP servers",
				Handler:     listMCPServersHandler(),
			},
		},
	}
}

// formatModelPresets renders the configured presets, marking the one the
// running agent resolved to.
func formatModelPresets(presets []ModelPreset) string {
	var b strings.Builder
	b.WriteString("Configured Models:")
	for _, preset := range presets {
		name := strings.TrimSpace(preset.Name)
		if name == "" {
			name = strings.TrimSpace(preset.Model)
		}
		b.WriteString("\n- " + name)

		target := strings.TrimSpace(preset.Model)
		if provider := strings.TrimSpace(preset.Provider); provider != "" {
			if target != "" {
				target = provider + "/" + target
			} else {
				target = provider
			}
		}
		if target != "" {
			b.WriteString(" (" + target + ")")
		}
		if preset.Current {
			b.WriteString(" - current")
		}
	}
	b.WriteString("\n\nTo change models, update config.json")
	return b.String()
}
