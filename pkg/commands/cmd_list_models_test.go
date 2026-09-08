package commands

import (
	"context"
	"strings"
	"testing"
)

func runListModels(t *testing.T, rt *Runtime) string {
	t.Helper()
	var got string
	req := Request{
		Text:  "/list models",
		Reply: func(msg string) error { got = msg; return nil },
	}
	res := NewExecutor(NewRegistry(BuiltinDefinitions()), rt).Execute(context.Background(), req)
	if res.Outcome != OutcomeHandled {
		t.Fatalf("outcome = %v, want handled", res.Outcome)
	}
	if res.Err != nil {
		t.Fatalf("handler error: %v", res.Err)
	}
	return got
}

func TestListModelsShowsEveryPresetAndMarksTheCurrentOne(t *testing.T) {
	rt := &Runtime{
		GetModelInfo: func() (string, string) { return "fast", "openai" },
		ListModelPresets: func() []ModelPreset {
			return []ModelPreset{
				{Name: "fast", Provider: "openai", Model: "gpt-4o-mini", Current: true},
				{Name: "deep", Provider: "anthropic", Model: "claude-sonnet"},
			}
		},
	}

	got := runListModels(t, rt)
	for _, want := range []string{"fast", "openai/gpt-4o-mini", "deep", "anthropic/claude-sonnet"} {
		if !strings.Contains(got, want) {
			t.Errorf("reply is missing %q:\n%s", want, got)
		}
	}
	currentLines := 0
	for _, line := range strings.Split(got, "\n") {
		if strings.Contains(line, "current") {
			currentLines++
			if !strings.Contains(line, "fast") {
				t.Errorf("the current marker is on the wrong preset: %q", line)
			}
		}
	}
	if currentLines != 1 {
		t.Errorf("current marker appears %d times, want exactly 1", currentLines)
	}
}

// No preset matches the running model: the list still renders, and nothing is
// marked rather than something being marked wrongly.
func TestListModelsWithNoCurrentMatch(t *testing.T) {
	rt := &Runtime{
		GetModelInfo: func() (string, string) { return "gone", "openai" },
		ListModelPresets: func() []ModelPreset {
			return []ModelPreset{{Name: "fast", Provider: "openai", Model: "gpt-4o-mini"}}
		},
	}

	got := runListModels(t, rt)
	if strings.Contains(got, "current") {
		t.Errorf("a preset was marked current with no match:\n%s", got)
	}
	if !strings.Contains(got, "fast") {
		t.Errorf("reply is missing the configured preset:\n%s", got)
	}
}

// An empty model_list, or a runtime that does not implement the callback, keeps
// the previous single-model reply.
func TestListModelsFallsBackToTheRunningModel(t *testing.T) {
	for name, rt := range map[string]*Runtime{
		"empty list": {
			GetModelInfo:     func() (string, string) { return "solo", "openai" },
			ListModelPresets: func() []ModelPreset { return nil },
		},
		"no callback": {
			GetModelInfo: func() (string, string) { return "solo", "openai" },
		},
	} {
		t.Run(name, func(t *testing.T) {
			got := runListModels(t, rt)
			if !strings.Contains(got, "Configured Model: solo") {
				t.Errorf("reply lost the previous format:\n%s", got)
			}
		})
	}
}

func TestListModelsWithoutAnyModelSourceReportsUnavailable(t *testing.T) {
	if got := runListModels(t, &Runtime{}); got != unavailableMsg {
		t.Errorf("reply = %q, want %q", got, unavailableMsg)
	}
}

// The formatter must not be able to print anything that is not in the DTO.
func TestListModelsNeverPrintsSecrets(t *testing.T) {
	rt := &Runtime{
		ListModelPresets: func() []ModelPreset {
			return []ModelPreset{{
				Name:     "fast",
				Provider: "openai",
				Model:    "gpt-4o-mini",
				Current:  true,
			}}
		},
	}

	got := runListModels(t, rt)
	for _, forbidden := range []string{"sk-", "api_key", "http://", "https://", "Authorization"} {
		if strings.Contains(got, forbidden) {
			t.Errorf("reply contains %q:\n%s", forbidden, got)
		}
	}
}

// A preset with no alias must still be identifiable.
func TestListModelsFallsBackToTheModelIdentifierWhenNameIsEmpty(t *testing.T) {
	rt := &Runtime{
		ListModelPresets: func() []ModelPreset {
			return []ModelPreset{{Provider: "openai", Model: "gpt-4o-mini"}}
		},
	}

	if got := runListModels(t, rt); !strings.Contains(got, "gpt-4o-mini") {
		t.Errorf("reply cannot identify an unnamed preset:\n%s", got)
	}
}
