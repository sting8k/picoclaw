package agent

import (
	"testing"

	"github.com/sipeed/picoclaw/pkg/config"
	"github.com/sipeed/picoclaw/pkg/providers"
)

func TestModelPresetsListsTheRunningConfig(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.ModelList = []*config.ModelConfig{
		{ModelName: "fast", Provider: "openai", Model: "gpt-4o-mini"},
		{ModelName: "deep", Model: "anthropic/claude-sonnet"},
	}
	agent := &AgentInstance{
		Model:      "deep",
		Candidates: []providers.FallbackCandidate{{Provider: "anthropic", Model: "claude-sonnet"}},
	}

	presets := modelPresets(cfg, agent)
	if len(presets) != 2 {
		t.Fatalf("presets = %d, want 2", len(presets))
	}
	if presets[0].Name != "fast" || presets[0].Provider != "openai" || presets[0].Model != "gpt-4o-mini" {
		t.Errorf("first preset = %+v", presets[0])
	}
	if presets[0].Current {
		t.Error("a preset that is not running was marked current")
	}
	// The second preset has no explicit provider: it comes from the model
	// identifier prefix, not from a guess, and the prefix is not repeated in
	// the model field.
	if presets[1].Provider != "anthropic" {
		t.Errorf("second preset provider = %q, want anthropic", presets[1].Provider)
	}
	if presets[1].Model != "claude-sonnet" {
		t.Errorf("second preset model = %q, want the identifier without its provider prefix", presets[1].Model)
	}
	if !presets[1].Current {
		t.Error("the running preset was not marked current")
	}
}

// Two presets can share an alias-free model name; only the one whose provider
// the agent resolved may be marked.
func TestModelPresetsMarksAtMostOneCurrent(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.ModelList = []*config.ModelConfig{
		{ModelName: "shared", Provider: "openai", Model: "gpt-4o-mini"},
		{ModelName: "shared", Provider: "azure", Model: "gpt-4o-mini"},
	}
	agent := &AgentInstance{
		Model:      "shared",
		Candidates: []providers.FallbackCandidate{{Provider: "azure", Model: "gpt-4o-mini"}},
	}

	presets := modelPresets(cfg, agent)
	marked := 0
	for _, preset := range presets {
		if preset.Current {
			marked++
			if preset.Provider != "azure" {
				t.Errorf("current marker on %q, want the resolved provider azure", preset.Provider)
			}
		}
	}
	if marked != 1 {
		t.Errorf("current markers = %d, want 1", marked)
	}
}

func TestModelPresetsIsEmptyWithoutAModelList(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.ModelList = nil
	if presets := modelPresets(cfg, &AgentInstance{Model: "x"}); presets != nil {
		t.Errorf("presets = %+v, want nil so the command keeps its old reply", presets)
	}
}

func TestModelPresetsSurvivesAMissingAgent(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.ModelList = []*config.ModelConfig{{ModelName: "fast", Provider: "openai", Model: "gpt-4o-mini"}}

	presets := modelPresets(cfg, nil)
	if len(presets) != 1 || presets[0].Current {
		t.Errorf("presets = %+v, want the list with nothing marked current", presets)
	}
}

// After the provider is created, the agent carries the resolved model id, not
// the alias from config. The preset must still be marked.
func TestModelPresetsMarksTheResolvedModelBehindAnAlias(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.ModelList = []*config.ModelConfig{
		{ModelName: "fast", Model: "openai/gpt-4o-mini"},
		{ModelName: "deep", Model: "anthropic/claude-sonnet"},
	}
	// This is what NewAgentLoop sees once CreateProvider has resolved the
	// preset: a concrete id in Model, and candidates carrying the identity.
	agent := &AgentInstance{
		Model: "gpt-4o-mini",
		Candidates: []providers.FallbackCandidate{
			{Provider: "openai", Model: "gpt-4o-mini", DisplayName: "fast"},
		},
	}

	presets := modelPresets(cfg, agent)
	if len(presets) != 2 {
		t.Fatalf("presets = %d, want 2", len(presets))
	}
	if !presets[0].Current {
		t.Error("the running preset was not marked when the agent carries the resolved id")
	}
	if presets[1].Current {
		t.Error("a preset that is not running was marked")
	}
	if presets[0].Provider != "openai" || presets[0].Model != "gpt-4o-mini" {
		t.Errorf("preset = %+v, want provider and model reported once each", presets[0])
	}
}

// The agent resolved a model id with no alias anywhere: the provider decides
// which of two identical ids is current.
func TestModelPresetsUsesTheProviderToDisambiguateResolvedIDs(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.ModelList = []*config.ModelConfig{
		{ModelName: "via-openai", Model: "openai/gpt-4o-mini"},
		{ModelName: "via-azure", Model: "azure/gpt-4o-mini"},
	}
	agent := &AgentInstance{
		Model: "gpt-4o-mini",
		Candidates: []providers.FallbackCandidate{
			{Provider: "azure", Model: "gpt-4o-mini"},
		},
	}

	presets := modelPresets(cfg, agent)
	marked := 0
	for _, preset := range presets {
		if preset.Current {
			marked++
			if preset.Name != "via-azure" {
				t.Errorf("current marker on %q, want the preset of the resolved provider", preset.Name)
			}
		}
	}
	if marked != 1 {
		t.Errorf("current markers = %d, want 1", marked)
	}
}

func TestSplitModelIdentifier(t *testing.T) {
	cases := map[string][2]string{
		"openai/gpt-4o-mini": {"openai", "gpt-4o-mini"},
		"gpt-4o-mini":        {"", "gpt-4o-mini"},
		"  openai/gpt  ":     {"openai", "gpt"},
		"":                   {"", ""},
	}
	for input, want := range cases {
		provider, model := splitModelIdentifier(input)
		if provider != want[0] || model != want[1] {
			t.Errorf("splitModelIdentifier(%q) = (%q, %q), want (%q, %q)",
				input, provider, model, want[0], want[1])
		}
	}
}

// Two presets can share one backend and differ only by alias. The alias the
// operator selected decides, not whichever appears first in the file.
func TestModelPresetsPrefersTheSelectedAliasOverAnEarlierTwin(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.ModelList = []*config.ModelConfig{
		{ModelName: "cheap", Model: "openai/gpt-4o-mini"},
		{ModelName: "chosen", Model: "openai/gpt-4o-mini"},
	}
	agent := &AgentInstance{
		Model: "chosen",
		Candidates: []providers.FallbackCandidate{
			{Provider: "openai", Model: "gpt-4o-mini", DisplayName: "chosen"},
		},
	}

	presets := modelPresets(cfg, agent)
	if presets[0].Current {
		t.Error("the earlier preset with the same backend was marked")
	}
	if !presets[1].Current {
		t.Errorf("the selected alias was not marked: %+v", presets)
	}
}

// The same, with the alias already resolved to a concrete id by the time the
// agent was built: identity is the only thing left to match on.
func TestModelPresetsFallsBackToIdentityWhenNoAliasMatches(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.ModelList = []*config.ModelConfig{
		{ModelName: "friendly-alias", Model: "openai/gpt-4o-mini"},
	}
	agent := &AgentInstance{
		Model:      "gpt-4o-mini",
		Candidates: []providers.FallbackCandidate{{Provider: "openai", Model: "gpt-4o-mini"}},
	}

	presets := modelPresets(cfg, agent)
	if len(presets) != 1 || !presets[0].Current {
		t.Errorf("the running preset was not marked through identity: %+v", presets)
	}
}
