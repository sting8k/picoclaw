package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sipeed/picoclaw/pkg/bus"
	"github.com/sipeed/picoclaw/pkg/config"
	"github.com/sipeed/picoclaw/pkg/media"
	"github.com/sipeed/picoclaw/pkg/providers"
	"github.com/sipeed/picoclaw/pkg/providers/protocoltypes"
	"github.com/sipeed/picoclaw/pkg/tools"
)

// scriptedProvider answers a fixed script and refuses to work after Close, so a
// turn that outlives its own generation fails loudly instead of silently
// succeeding.
type scriptedProvider struct {
	name    string
	replies []scriptedReply

	mu     sync.Mutex
	calls  int
	closes int
	closed bool
	used   bool // Chat was called after Close
}

type scriptedReply struct {
	text        string
	toolCommand string
}

var errProviderClosed = errors.New("provider is closed")

func (p *scriptedProvider) Chat(
	_ context.Context,
	_ []providers.Message,
	_ []providers.ToolDefinition,
	_ string,
	_ map[string]any,
) (*providers.LLMResponse, error) {
	p.mu.Lock()
	if p.closed {
		p.used = true
		p.mu.Unlock()
		return nil, errProviderClosed
	}
	idx := p.calls
	p.calls++
	var reply scriptedReply
	if idx < len(p.replies) {
		reply = p.replies[idx]
	} else {
		reply = scriptedReply{text: p.name + ": out of script"}
	}
	p.mu.Unlock()

	if reply.toolCommand != "" {
		args, _ := json.Marshal(map[string]any{"action": "run", "command": reply.toolCommand})
		return &providers.LLMResponse{
			FinishReason: "tool_calls",
			ToolCalls: []providers.ToolCall{{
				ID:   fmt.Sprintf("%s-%d", p.name, idx),
				Type: "function",
				Function: &protocoltypes.FunctionCall{
					Name:      "exec",
					Arguments: string(args),
				},
			}},
		}, nil
	}
	return &providers.LLMResponse{Content: reply.text, FinishReason: "stop"}, nil
}

func (p *scriptedProvider) GetDefaultModel() string { return "scripted" }

func (p *scriptedProvider) Close() {
	p.mu.Lock()
	p.closes++
	p.closed = true
	p.mu.Unlock()
}

func (p *scriptedProvider) stats() (calls, closes int, usedAfterClose bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.calls, p.closes, p.used
}

// reloadTestConfig is a config with the exec tool on, so a turn can be pinned
// inside a tool call between two LLM calls.
func reloadTestConfig(t *testing.T, workspace string) *config.Config {
	t.Helper()
	cfg := config.DefaultConfig()
	cfg.Agents.Defaults.Workspace = workspace
	cfg.Agents.Defaults.RestrictToWorkspace = false
	cfg.Agents.Defaults.MaxToolIterations = 5
	cfg.ModelList = nil
	cfg.Tools.Exec.Enabled = true
	cfg.Hooks.Enabled = false
	cfg.Tools.MCP.Enabled = false
	return cfg
}

// fileBarrier pins a turn inside its tool call until the test releases it.
type fileBarrier struct {
	ready  string
	unlock string
}

func newFileBarrier(t *testing.T) *fileBarrier {
	t.Helper()
	dir := t.TempDir()
	return &fileBarrier{ready: filepath.Join(dir, "ready"), unlock: filepath.Join(dir, "unlock")}
}

func (b *fileBarrier) command() string {
	return fmt.Sprintf("touch %q; while [ ! -f %q ]; do sleep 0.02; done; echo released", b.ready, b.unlock)
}

func (b *fileBarrier) waitEntered(t *testing.T) {
	t.Helper()
	waitUntil(t, 10*time.Second, "the tool to start", func() bool {
		_, err := os.Stat(b.ready)
		return err == nil
	})
}

func (b *fileBarrier) release(t *testing.T) {
	t.Helper()
	if err := os.WriteFile(b.unlock, []byte("go"), 0o644); err != nil {
		t.Fatal(err)
	}
}

func waitUntil(t *testing.T, timeout time.Duration, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("timed out after %s waiting for %s", timeout, what)
}

func inboundMessage(chatID, content string) bus.InboundMessage {
	return bus.InboundMessage{
		Context: bus.InboundContext{
			Channel:  "cli",
			ChatID:   chatID,
			ChatType: "direct",
			SenderID: "tester",
		},
		Content: content,
	}
}

// A candidate whose hooks cannot start must not become the running generation.
func TestReloadRejectsCandidateWithFailingHooks(t *testing.T) {
	workspace := t.TempDir()
	cfg := reloadTestConfig(t, workspace)
	old := &scriptedProvider{name: "old", replies: []scriptedReply{{text: "old answers"}}}

	al := NewAgentLoop(cfg, bus.NewMessageBus(), old)
	defer al.Close()

	candidate := reloadTestConfig(t, workspace)
	candidate.Hooks.Enabled = true
	candidate.Hooks.Processes = map[string]config.ProcessHookConfig{
		"broken": {Enabled: true, Command: []string{filepath.Join(workspace, "no-such-hook-binary")}},
	}

	next := &scriptedProvider{name: "next"}
	err := al.ReloadProviderAndConfig(context.Background(), next, candidate)
	if err == nil {
		t.Fatal("ReloadProviderAndConfig() = nil, want an error for a candidate whose hooks cannot start")
	}
	if !strings.Contains(err.Error(), "hooks") {
		t.Errorf("error = %v, want it to name hooks", err)
	}

	if al.GetConfig() != cfg {
		t.Error("the failed candidate config became the running one")
	}
	if _, closes, _ := old.stats(); closes != 0 {
		t.Errorf("old provider Close() calls = %d, want 0", closes)
	}

	// The old generation must still serve a turn.
	if _, err := al.processMessage(context.Background(), inboundMessage("chat", "hello")); err != nil {
		t.Fatalf("turn after the failed reload: %v", err)
	}
	calls, _, usedAfterClose := old.stats()
	if calls == 0 {
		t.Error("the old provider served no turn after the failed reload")
	}
	if usedAfterClose {
		t.Error("the old provider was used after being closed")
	}
}

// Same contract for MCP, which fails on a different code path than hooks.
func TestReloadRejectsCandidateWithFailingMCP(t *testing.T) {
	workspace := t.TempDir()
	cfg := reloadTestConfig(t, workspace)
	old := &scriptedProvider{name: "old", replies: []scriptedReply{{text: "old answers"}}}

	al := NewAgentLoop(cfg, bus.NewMessageBus(), old)
	defer al.Close()

	candidate := reloadTestConfig(t, workspace)
	candidate.Tools.MCP.Enabled = true
	candidate.Tools.MCP.Servers = map[string]config.MCPServerConfig{
		"broken": {Enabled: true, Command: filepath.Join(workspace, "no-such-mcp-binary")},
	}

	next := &scriptedProvider{name: "next"}
	err := al.ReloadProviderAndConfig(context.Background(), next, candidate)
	if err == nil {
		t.Fatal("ReloadProviderAndConfig() = nil, want an error for a candidate whose MCP servers cannot start")
	}
	if !strings.Contains(err.Error(), "mcp") {
		t.Errorf("error = %v, want it to name mcp", err)
	}

	if al.GetConfig() != cfg {
		t.Error("the failed candidate config became the running one")
	}
	if al.mcp.hasManager() {
		t.Error("a failed candidate left an MCP manager behind on the running loop")
	}
	if al.mcp.getInitErr() != nil {
		t.Errorf("the running loop inherited the candidate's MCP error: %v", al.mcp.getInitErr())
	}
	if _, closes, _ := old.stats(); closes != 0 {
		t.Errorf("old provider Close() calls = %d, want 0", closes)
	}
	if _, err := al.processMessage(context.Background(), inboundMessage("chat", "hello")); err != nil {
		t.Fatalf("turn after the failed reload: %v", err)
	}
}

// The reload must wait for a whole turn, not for the LLM request that happens
// to be in flight. A turn parked inside a tool holds no request, and its next
// LLM call still belongs to the old provider.
func TestReloadWaitsForATurnParkedInsideAToolCall(t *testing.T) {
	workspace := t.TempDir()
	cfg := reloadTestConfig(t, workspace)
	barrier := newFileBarrier(t)

	old := &scriptedProvider{name: "old", replies: []scriptedReply{
		{toolCommand: barrier.command()},
		{text: "old finished"},
	}}
	al := NewAgentLoop(cfg, bus.NewMessageBus(), old)
	defer al.Close()

	turnDone := make(chan error, 1)
	go func() {
		_, err := al.processMessage(context.Background(), inboundMessage("chat", "long turn"))
		turnDone <- err
	}()
	barrier.waitEntered(t)

	reloadDone := make(chan error, 1)
	next := &scriptedProvider{name: "next", replies: []scriptedReply{{text: "next answers"}}}
	go func() {
		reloadDone <- al.ReloadProviderAndConfig(context.Background(), next, reloadTestConfig(t, workspace))
	}()

	select {
	case err := <-reloadDone:
		t.Fatalf("reload returned while a turn was still inside a tool call: %v", err)
	case <-time.After(200 * time.Millisecond):
	}
	if _, closes, _ := old.stats(); closes != 0 {
		t.Fatal("the old provider was closed while a turn was still using it")
	}

	barrier.release(t)
	if err := <-turnDone; err != nil {
		t.Fatalf("the old turn failed: %v", err)
	}
	select {
	case err := <-reloadDone:
		if err != nil {
			t.Fatalf("reload failed: %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("reload never returned after the turn finished")
	}

	calls, closes, usedAfterClose := old.stats()
	if calls != 2 {
		t.Errorf("old provider served %d calls, want both calls of the old turn", calls)
	}
	if usedAfterClose {
		t.Error("the old turn called its provider after the reload closed it")
	}
	if closes != 1 {
		t.Errorf("old provider Close() calls = %d, want exactly 1", closes)
	}
	if al.GetConfig() == cfg {
		t.Error("the reload did not install the new config")
	}
}

// A message admitted while a reload is committing must run on exactly one
// generation, and must not be dropped.
func TestTurnsAdmittedDuringReloadRunOnTheNewGeneration(t *testing.T) {
	workspace := t.TempDir()
	cfg := reloadTestConfig(t, workspace)
	cfg.Agents.Defaults.MaxParallelTurns = 2
	barrier := newFileBarrier(t)

	old := &scriptedProvider{name: "old", replies: []scriptedReply{
		{toolCommand: barrier.command()},
		{text: "old finished"},
	}}
	al := NewAgentLoop(cfg, bus.NewMessageBus(), old)
	defer al.Close()

	turnDone := make(chan error, 1)
	go func() {
		_, err := al.processMessage(context.Background(), inboundMessage("chat-a", "long turn"))
		turnDone <- err
	}()
	barrier.waitEntered(t)

	next := &scriptedProvider{name: "next", replies: []scriptedReply{{text: "next answers"}}}
	reloadDone := make(chan error, 1)
	go func() {
		reloadDone <- al.ReloadProviderAndConfig(context.Background(), next, reloadTestConfig(t, workspace))
	}()
	waitUntil(t, 5*time.Second, "the reload to close admission", func() bool {
		al.turns.mu.Lock()
		defer al.turns.mu.Unlock()
		return al.turns.blocked
	})

	// This one arrives while admission is closed. It must wait, not fail.
	parked := make(chan error, 1)
	go func() {
		_, err := al.processMessage(context.Background(), inboundMessage("chat-b", "arrives during the reload"))
		parked <- err
	}()
	select {
	case err := <-parked:
		t.Fatalf("a turn ran while the reload held admission: %v", err)
	case <-time.After(150 * time.Millisecond):
	}

	barrier.release(t)
	if err := <-turnDone; err != nil {
		t.Fatalf("the old turn failed: %v", err)
	}
	if err := <-reloadDone; err != nil {
		t.Fatalf("reload failed: %v", err)
	}
	select {
	case err := <-parked:
		if err != nil {
			t.Fatalf("the parked turn failed: %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("the parked turn never ran after the reload finished")
	}

	if calls, _, _ := next.stats(); calls != 1 {
		t.Errorf("new provider served %d calls, want the parked turn", calls)
	}
	if _, _, usedAfterClose := old.stats(); usedAfterClose {
		t.Error("the parked turn used the closed old provider")
	}
}

// Cancelling the reload before it commits must leave the loop serving.
func TestCanceledReloadReopensAdmission(t *testing.T) {
	workspace := t.TempDir()
	cfg := reloadTestConfig(t, workspace)
	barrier := newFileBarrier(t)

	old := &scriptedProvider{name: "old", replies: []scriptedReply{
		{toolCommand: barrier.command()},
		{text: "old finished"},
		{text: "old still serving"},
	}}
	al := NewAgentLoop(cfg, bus.NewMessageBus(), old)
	defer al.Close()

	turnDone := make(chan error, 1)
	go func() {
		_, err := al.processMessage(context.Background(), inboundMessage("chat", "long turn"))
		turnDone <- err
	}()
	barrier.waitEntered(t)

	ctx, cancel := context.WithCancel(context.Background())
	reloadDone := make(chan error, 1)
	next := &scriptedProvider{name: "next"}
	go func() {
		reloadDone <- al.ReloadProviderAndConfig(ctx, next, reloadTestConfig(t, workspace))
	}()
	waitUntil(t, 5*time.Second, "the reload to close admission", func() bool {
		al.turns.mu.Lock()
		defer al.turns.mu.Unlock()
		return al.turns.blocked
	})
	cancel()

	select {
	case err := <-reloadDone:
		if err == nil {
			t.Fatal("a canceled reload reported success")
		}
		if !errors.Is(err, context.Canceled) {
			t.Errorf("error = %v, want it to wrap context.Canceled", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("the canceled reload never returned")
	}

	barrier.release(t)
	if err := <-turnDone; err != nil {
		t.Fatalf("the running turn failed after the canceled reload: %v", err)
	}
	if al.GetConfig() != cfg {
		t.Error("a canceled reload changed the running config")
	}
	if _, closes, _ := old.stats(); closes != 0 {
		t.Errorf("a canceled reload closed the old provider (%d times)", closes)
	}
	if _, err := al.processMessage(context.Background(), inboundMessage("chat", "still there?")); err != nil {
		t.Fatalf("the loop stopped serving after a canceled reload: %v", err)
	}
}

// Repeated reloads must retire each generation exactly once.
func TestRepeatedReloadsCloseEachProviderOnce(t *testing.T) {
	workspace := t.TempDir()
	cfg := reloadTestConfig(t, workspace)

	first := &scriptedProvider{name: "gen-0", replies: []scriptedReply{{text: "hi"}}}
	al := NewAgentLoop(cfg, bus.NewMessageBus(), first)
	defer al.Close()

	generations := []*scriptedProvider{first}
	for i := 1; i <= 3; i++ {
		next := &scriptedProvider{name: fmt.Sprintf("gen-%d", i), replies: []scriptedReply{{text: "hi"}}}
		if err := al.ReloadProviderAndConfig(context.Background(), next, reloadTestConfig(t, workspace)); err != nil {
			t.Fatalf("reload %d: %v", i, err)
		}
		if _, err := al.processMessage(context.Background(), inboundMessage("chat", "hello")); err != nil {
			t.Fatalf("turn after reload %d: %v", i, err)
		}
		generations = append(generations, next)
	}

	for i, gen := range generations {
		calls, closes, usedAfterClose := gen.stats()
		want := 1
		if i == len(generations)-1 {
			want = 0 // the live one is not closed by a reload
		}
		if closes != want {
			t.Errorf("%s Close() calls = %d, want %d", gen.name, closes, want)
		}
		if usedAfterClose {
			t.Errorf("%s was used after being closed", gen.name)
		}
		if i > 0 && calls == 0 {
			t.Errorf("%s never served a turn", gen.name)
		}
	}
}

// Two reloads at once: one wins, the other is refused, and the loop keeps
// serving either way.
func TestConcurrentReloadIsRefusedNotInterleaved(t *testing.T) {
	workspace := t.TempDir()
	cfg := reloadTestConfig(t, workspace)
	barrier := newFileBarrier(t)

	old := &scriptedProvider{name: "old", replies: []scriptedReply{
		{toolCommand: barrier.command()},
		{text: "old finished"},
	}}
	al := NewAgentLoop(cfg, bus.NewMessageBus(), old)
	defer al.Close()

	turnDone := make(chan error, 1)
	go func() {
		_, err := al.processMessage(context.Background(), inboundMessage("chat", "long turn"))
		turnDone <- err
	}()
	barrier.waitEntered(t)

	firstDone := make(chan error, 1)
	go func() {
		firstDone <- al.ReloadProviderAndConfig(
			context.Background(),
			&scriptedProvider{name: "first", replies: []scriptedReply{{text: "hi"}}},
			reloadTestConfig(t, workspace),
		)
	}()
	waitUntil(t, 5*time.Second, "the first reload to close admission", func() bool {
		al.turns.mu.Lock()
		defer al.turns.mu.Unlock()
		return al.turns.blocked
	})

	second := al.ReloadProviderAndConfig(
		context.Background(),
		&scriptedProvider{name: "second"},
		reloadTestConfig(t, workspace),
	)
	if second == nil {
		t.Fatal("a second concurrent reload reported success")
	}

	barrier.release(t)
	if err := <-turnDone; err != nil {
		t.Fatalf("the running turn failed: %v", err)
	}
	if err := <-firstDone; err != nil {
		t.Fatalf("the first reload failed: %v", err)
	}
	if _, err := al.processMessage(context.Background(), inboundMessage("chat", "still there?")); err != nil {
		t.Fatalf("the loop stopped serving after a refused concurrent reload: %v", err)
	}
}

// A refused second reload must not touch the barrier the first one owns.
// Getting this wrong reopens admission in the middle of a commit, which is
// worse than the bug the barrier exists to fix.
func TestRefusedConcurrentReloadLeavesTheBarrierClosed(t *testing.T) {
	workspace := t.TempDir()
	cfg := reloadTestConfig(t, workspace)
	barrier := newFileBarrier(t)

	old := &scriptedProvider{name: "old", replies: []scriptedReply{
		{toolCommand: barrier.command()},
		{text: "old finished"},
	}}
	al := NewAgentLoop(cfg, bus.NewMessageBus(), old)
	defer al.Close()

	turnDone := make(chan error, 1)
	go func() {
		_, err := al.processMessage(context.Background(), inboundMessage("chat", "long turn"))
		turnDone <- err
	}()
	barrier.waitEntered(t)

	firstDone := make(chan error, 1)
	go func() {
		firstDone <- al.ReloadProviderAndConfig(
			context.Background(),
			&scriptedProvider{name: "first", replies: []scriptedReply{{text: "hi"}}},
			reloadTestConfig(t, workspace),
		)
	}()
	waitUntil(t, 5*time.Second, "the first reload to close admission", func() bool {
		al.turns.mu.Lock()
		defer al.turns.mu.Unlock()
		return al.turns.blocked
	})

	second := al.ReloadProviderAndConfig(
		context.Background(),
		&scriptedProvider{name: "second"},
		reloadTestConfig(t, workspace),
	)
	if !errors.Is(second, ErrReloadInProgress) {
		t.Fatalf("second reload error = %v, want ErrReloadInProgress", second)
	}

	al.turns.mu.Lock()
	stillBlocked := al.turns.blocked
	al.turns.mu.Unlock()
	if !stillBlocked {
		t.Fatal("the refused reload reopened the barrier the first one owns")
	}

	// A new turn must still be held back, and a third reload still refused.
	admitted := make(chan struct{})
	go func() {
		_, _ = al.processMessage(context.Background(), inboundMessage("chat-b", "during the reload"))
		close(admitted)
	}()
	select {
	case <-admitted:
		t.Fatal("a turn was admitted after the refused reload")
	case <-time.After(150 * time.Millisecond):
	}
	if third := al.ReloadProviderAndConfig(
		context.Background(),
		&scriptedProvider{name: "third"},
		reloadTestConfig(t, workspace),
	); !errors.Is(third, ErrReloadInProgress) {
		t.Errorf("third reload error = %v, want ErrReloadInProgress", third)
	}

	barrier.release(t)
	if err := <-turnDone; err != nil {
		t.Fatalf("the running turn failed: %v", err)
	}
	if err := <-firstDone; err != nil {
		t.Fatalf("the first reload failed: %v", err)
	}
	<-admitted
}

// A steering continuation runs a full turn. It must wait for a generation
// change like any other turn, and the queue must survive the wait.
func TestSteeringContinuationWaitsForTheBarrier(t *testing.T) {
	workspace := t.TempDir()
	cfg := reloadTestConfig(t, workspace)
	provider := &scriptedProvider{name: "gen-0", replies: []scriptedReply{{text: "answered"}}}

	al := NewAgentLoop(cfg, bus.NewMessageBus(), provider)
	defer al.Close()

	// One turn so the session exists, then queue steering for it.
	if _, err := al.processMessage(context.Background(), inboundMessage("chat", "first")); err != nil {
		t.Fatalf("first turn: %v", err)
	}
	sessionKey, agentID, ok := al.resolveSteeringTarget(inboundMessage("chat", "steer"))
	if !ok {
		t.Fatal("could not resolve the steering target")
	}
	if err := al.enqueueSteeringMessage(sessionKey, agentID, providers.Message{
		Role: "user", Content: "steer me",
	}); err != nil {
		t.Fatalf("enqueue: %v", err)
	}

	// Hold the barrier the way a reload does.
	if err := al.turns.block(context.Background(), 5*time.Second); err != nil {
		t.Fatalf("block: %v", err)
	}

	callsBefore, _, _ := provider.stats()
	continued := make(chan error, 1)
	go func() {
		_, err := al.Continue(context.Background(), sessionKey, "cli", "chat")
		continued <- err
	}()

	select {
	case err := <-continued:
		t.Fatalf("a continuation ran while a generation change held the barrier: %v", err)
	case <-time.After(200 * time.Millisecond):
	}
	if calls, _, _ := provider.stats(); calls != callsBefore {
		t.Fatalf("the continuation called the provider while the barrier was held (%d -> %d)",
			callsBefore, calls)
	}
	if al.pendingSteeringCountForScope(sessionKey) == 0 {
		t.Error("the queued steering message was consumed while the barrier was held")
	}

	al.turns.unblock()
	select {
	case err := <-continued:
		if err != nil {
			t.Fatalf("the continuation failed after the barrier reopened: %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("the continuation never ran after the barrier reopened")
	}
	if calls, _, _ := provider.stats(); calls == callsBefore {
		t.Error("the continuation never reached the provider")
	}
}

// A cancelled continuation must leave the queue intact for the next turn.
func TestCanceledContinuationKeepsTheSteeringQueue(t *testing.T) {
	workspace := t.TempDir()
	cfg := reloadTestConfig(t, workspace)
	provider := &scriptedProvider{name: "gen-0", replies: []scriptedReply{{text: "answered"}}}

	al := NewAgentLoop(cfg, bus.NewMessageBus(), provider)
	defer al.Close()

	if _, err := al.processMessage(context.Background(), inboundMessage("chat", "first")); err != nil {
		t.Fatalf("first turn: %v", err)
	}
	sessionKey, agentID, ok := al.resolveSteeringTarget(inboundMessage("chat", "steer"))
	if !ok {
		t.Fatal("could not resolve the steering target")
	}
	if err := al.enqueueSteeringMessage(sessionKey, agentID, providers.Message{
		Role: "user", Content: "steer me",
	}); err != nil {
		t.Fatalf("enqueue: %v", err)
	}

	if err := al.turns.block(context.Background(), 5*time.Second); err != nil {
		t.Fatalf("block: %v", err)
	}
	defer al.turns.unblock()

	ctx, cancel := context.WithCancel(context.Background())
	continued := make(chan error, 1)
	go func() {
		_, err := al.Continue(ctx, sessionKey, "cli", "chat")
		continued <- err
	}()
	time.Sleep(100 * time.Millisecond)
	cancel()

	select {
	case err := <-continued:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("error = %v, want context.Canceled", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("the canceled continuation never returned")
	}
	if al.pendingSteeringCountForScope(sessionKey) == 0 {
		t.Error("a canceled continuation dropped the queued steering message")
	}
	if _, ok := al.activeTurnStates.Load(sessionKey); ok {
		t.Error("a canceled continuation left the session claimed")
	}
}

// Direct turns bring hooks and MCP up themselves; that has to happen under the
// barrier too, or a commit can land in the middle of it.
func TestDirectTurnInitializesSubsystemsUnderTheBarrier(t *testing.T) {
	workspace := t.TempDir()
	cfg := reloadTestConfig(t, workspace)
	provider := &scriptedProvider{name: "gen-0", replies: []scriptedReply{{text: "answered"}}}

	al := NewAgentLoop(cfg, bus.NewMessageBus(), provider)
	defer al.Close()

	if err := al.turns.block(context.Background(), 5*time.Second); err != nil {
		t.Fatalf("block: %v", err)
	}

	done := make(chan error, 1)
	go func() {
		_, err := al.ProcessDirectWithChannel(context.Background(), "hello", "direct-session", "cli", "direct")
		done <- err
	}()
	select {
	case err := <-done:
		t.Fatalf("a direct turn ran while the barrier was held: %v", err)
	case <-time.After(150 * time.Millisecond):
	}

	al.turns.unblock()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("the direct turn failed: %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("the direct turn never ran after the barrier reopened")
	}
}

// The drain timeout must be recognizable, because it is the one reload failure
// a caller can retry unchanged.
func TestDrainTimeoutIsDistinguishable(t *testing.T) {
	workspace := t.TempDir()
	cfg := reloadTestConfig(t, workspace)
	barrier := newFileBarrier(t)

	old := &scriptedProvider{name: "old", replies: []scriptedReply{
		{toolCommand: barrier.command()},
		{text: "old finished"},
	}}
	al := NewAgentLoop(cfg, bus.NewMessageBus(), old)
	defer al.Close()

	turnDone := make(chan error, 1)
	go func() {
		_, err := al.processMessage(context.Background(), inboundMessage("chat", "long turn"))
		turnDone <- err
	}()
	barrier.waitEntered(t)

	err := al.turns.block(context.Background(), 50*time.Millisecond)
	if !errors.Is(err, ErrTurnsBusy) {
		t.Fatalf("block error = %v, want ErrTurnsBusy", err)
	}
	al.turns.mu.Lock()
	blocked := al.turns.blocked
	al.turns.mu.Unlock()
	if blocked {
		t.Error("a timed-out block left admission closed")
	}

	barrier.release(t)
	if err := <-turnDone; err != nil {
		t.Fatalf("the running turn failed: %v", err)
	}
}

// A background sub-turn outlives the turn that spawned it, so draining turns
// does not cover it.
//
// The window this closes: the child pins its agent instance early
// (subturn.go, before the parent-child link), so registering it any later
// leaves a moment where a reload sees nothing running, commits, closes the old
// provider - and the child then calls it. The parent's mutex is held here to
// park the child in exactly that moment.
func TestReloadWaitsForABackgroundSubTurnPinnedBeforeItRegisters(t *testing.T) {
	workspace := t.TempDir()
	cfg := reloadTestConfig(t, workspace)
	provider := &scriptedProvider{name: "gen-0", replies: []scriptedReply{{text: "child answer"}}}

	al := NewAgentLoop(cfg, bus.NewMessageBus(), provider)
	defer al.Close()

	// A parent that has already finished its own turn: it holds no barrier, so
	// only the child's own registration can hold the reload back.
	parent := &turnState{
		ctx:            context.Background(),
		turnID:         "parent-1",
		childTurnIDs:   []string{},
		pendingResults: make(chan *tools.ToolResult, 4),
		session:        &ephemeralSessionStore{},
		agent:          al.registry.GetDefaultAgent(),
	}

	// Park the child where it has pinned the generation but done nothing else.
	parent.mu.Lock()
	spawned := make(chan error, 1)
	go func() {
		_, err := spawnSubTurn(context.Background(), al, parent, SubTurnConfig{
			Model:    "gpt-4o-mini",
			Tools:    []tools.Tool{},
			Critical: true,
		})
		spawned <- err
	}()
	waitUntil(t, 10*time.Second, "the sub-turn to reach the parent link", func() bool {
		parked := false
		al.activeTurnStates.Range(func(key, _ any) bool {
			if name, ok := key.(string); ok && strings.HasPrefix(name, "subturn-") {
				parked = true
			}
			return !parked
		})
		return parked
	})

	restore := turnDrainGracePeriod
	turnDrainGracePeriod = 300 * time.Millisecond
	defer func() { turnDrainGracePeriod = restore }()

	err := al.ReloadProviderAndConfig(
		context.Background(),
		&scriptedProvider{name: "candidate"},
		reloadTestConfig(t, workspace),
	)
	if !errors.Is(err, ErrTurnsBusy) {
		t.Fatalf("reload error = %v, want ErrTurnsBusy while a sub-turn is running", err)
	}
	if al.GetConfig() != cfg {
		t.Error("a refused reload changed the running config")
	}
	if _, closes, _ := provider.stats(); closes != 0 {
		t.Errorf("a refused reload closed the provider the sub-turn is about to use (%d times)", closes)
	}

	parent.mu.Unlock()
	if err := <-spawned; err != nil {
		t.Fatalf("the sub-turn failed: %v", err)
	}
	if _, _, usedAfterClose := provider.stats(); usedAfterClose {
		t.Error("the sub-turn called a provider that had already been closed")
	}

	// With the sub-turn finished the same reload succeeds.
	if err := al.ReloadProviderAndConfig(
		context.Background(),
		&scriptedProvider{name: "candidate", replies: []scriptedReply{{text: "hi"}}},
		reloadTestConfig(t, workspace),
	); err != nil {
		t.Fatalf("reload after the sub-turn finished: %v", err)
	}
}

// A sub-turn that starts while a commit is under way must be refused, not
// parked: waiting would only let it pin the generation being retired.
func TestSubTurnRefusedWhileACommitIsUnderWay(t *testing.T) {
	workspace := t.TempDir()
	cfg := reloadTestConfig(t, workspace)
	provider := &scriptedProvider{name: "gen-0", replies: []scriptedReply{{text: "hi"}}}

	al := NewAgentLoop(cfg, bus.NewMessageBus(), provider)
	defer al.Close()

	// Admission closed with nothing running: exactly the window a commit runs in.
	if err := al.turns.block(context.Background(), 5*time.Second); err != nil {
		t.Fatalf("block: %v", err)
	}
	defer al.turns.unblock()

	parent := &turnState{
		ctx:            context.Background(),
		turnID:         "parent-1",
		childTurnIDs:   []string{},
		pendingResults: make(chan *tools.ToolResult, 4),
		session:        &ephemeralSessionStore{},
		agent:          al.registry.GetDefaultAgent(),
	}
	_, err := spawnSubTurn(context.Background(), al, parent, SubTurnConfig{
		Model: "gpt-4o-mini",
		Tools: []tools.Tool{},
	})
	if !errors.Is(err, ErrReloadInProgress) {
		t.Fatalf("spawnSubTurn error = %v, want ErrReloadInProgress", err)
	}
	if calls, _, _ := provider.stats(); calls != 0 {
		t.Errorf("the refused sub-turn still called the provider %d times", calls)
	}
}

// countingHook records every BeforeLLM it sees, so a test can tell "still
// mounted" from "mounted but dead".
type countingHook struct{ calls atomic.Int64 }

func (h *countingHook) BeforeLLM(
	_ context.Context,
	req *LLMHookRequest,
) (*LLMHookRequest, HookDecision, error) {
	h.calls.Add(1)
	return req, HookDecision{}, nil
}

func (h *countingHook) AfterLLM(
	_ context.Context,
	resp *LLMHookResponse,
) (*LLMHookResponse, HookDecision, error) {
	return resp, HookDecision{}, nil
}

// A reload replaces the hooks that came from config. A hook someone mounted
// through the public API is not one of those, and unregistering it would be a
// silent regression for any embedder.
func TestReloadKeepsExternallyMountedHooks(t *testing.T) {
	workspace := t.TempDir()
	cfg := reloadTestConfig(t, workspace)
	provider := &scriptedProvider{name: "gen-0", replies: []scriptedReply{{text: "before"}, {text: "after"}}}

	al := NewAgentLoop(cfg, bus.NewMessageBus(), provider)
	defer al.Close()

	external := &countingHook{}
	if err := al.MountHook(NamedHook("external-policy", external)); err != nil {
		t.Fatalf("MountHook: %v", err)
	}
	if _, err := al.processMessage(context.Background(), inboundMessage("chat", "before")); err != nil {
		t.Fatalf("first turn: %v", err)
	}
	before := external.calls.Load()
	if before == 0 {
		t.Fatal("the external hook never ran before the reload")
	}

	next := &scriptedProvider{name: "gen-1", replies: []scriptedReply{{text: "after"}}}
	if err := al.ReloadProviderAndConfig(context.Background(), next, reloadTestConfig(t, workspace)); err != nil {
		t.Fatalf("reload: %v", err)
	}

	if _, err := al.processMessage(context.Background(), inboundMessage("chat", "after")); err != nil {
		t.Fatalf("turn after the reload: %v", err)
	}
	if after := external.calls.Load(); after <= before {
		t.Errorf("the external hook stopped running after the reload (%d -> %d)", before, after)
	}
}

// The in-memory session fallback appears when the on-disk store cannot be
// created. Swapping generations on it drops the conversation, so the reload
// refuses and points at what to repair instead of reporting success.
func TestReloadRefusesTheInMemorySessionFallback(t *testing.T) {
	workspace := t.TempDir()
	// A file where the sessions directory belongs: the JSONL store cannot be
	// created and instance.go falls back to the in-memory manager.
	if err := os.WriteFile(filepath.Join(workspace, "sessions"), []byte("not a dir"), 0o644); err != nil {
		t.Fatal(err)
	}

	cfg := reloadTestConfig(t, workspace)
	provider := &scriptedProvider{name: "gen-0", replies: []scriptedReply{{text: "answered"}}}
	al := NewAgentLoop(cfg, bus.NewMessageBus(), provider)
	defer al.Close()

	if _, ok := nonPersistentSessionStore(al.GetRegistry()); !ok {
		t.Skip("the on-disk session store started anyway; this path needs the fallback")
	}

	err := al.ReloadProviderAndConfig(
		context.Background(),
		&scriptedProvider{name: "candidate"},
		reloadTestConfig(t, workspace),
	)
	if !errors.Is(err, ErrSessionsNotPersistent) {
		t.Fatalf("reload error = %v, want ErrSessionsNotPersistent", err)
	}
	if !strings.Contains(err.Error(), "sessions directory") {
		t.Errorf("error does not say what to repair: %v", err)
	}
	if al.GetConfig() != cfg {
		t.Error("a refused reload changed the running config")
	}
	if _, closes, _ := provider.stats(); closes != 0 {
		t.Errorf("a refused reload closed the old provider (%d times)", closes)
	}
	// Turns are not asserted here: this workspace cannot persist sessions at
	// all, which is exactly the condition the refusal reports.
}

// An async spawn captures its parent turn and may reach spawnSubTurn long
// after that turn ended - and after a reload retired the agent instance and
// provider it captured. Counting what is running cannot see that: nothing is
// running. The parent's generation is what makes it visible.
func TestSubTurnOfARetiredParentIsRefused(t *testing.T) {
	workspace := t.TempDir()
	cfg := reloadTestConfig(t, workspace)
	old := &scriptedProvider{name: "old", replies: []scriptedReply{{text: "child answer"}}}

	al := NewAgentLoop(cfg, bus.NewMessageBus(), old)
	defer al.Close()

	// A finished turn from the generation that is about to be replaced.
	parent := &turnState{
		ctx:            context.Background(),
		turnID:         "finished-parent",
		generation:     al.turns.currentGeneration(),
		childTurnIDs:   []string{},
		pendingResults: make(chan *tools.ToolResult, 4),
		session:        &ephemeralSessionStore{},
		agent:          al.registry.GetDefaultAgent(),
	}

	if err := al.ReloadProviderAndConfig(
		context.Background(),
		&scriptedProvider{name: "new", replies: []scriptedReply{{text: "hi"}}},
		reloadTestConfig(t, workspace),
	); err != nil {
		t.Fatalf("reload: %v", err)
	}

	_, err := spawnSubTurn(context.Background(), al, parent, SubTurnConfig{
		Model:    "gpt-4o-mini",
		Tools:    []tools.Tool{},
		Critical: true,
	})
	if !errors.Is(err, ErrStaleGeneration) {
		t.Fatalf("spawnSubTurn error = %v, want ErrStaleGeneration", err)
	}
	if calls, _, usedAfterClose := old.stats(); usedAfterClose || calls != 0 {
		t.Errorf("the late child used the retired generation (calls=%d, after close=%v)",
			calls, usedAfterClose)
	}
}

// The same check must not get in the way of a sub-turn spawned by a turn that
// is still running on the live generation.
func TestSubTurnOfALiveParentStillRuns(t *testing.T) {
	workspace := t.TempDir()
	cfg := reloadTestConfig(t, workspace)
	first := &scriptedProvider{name: "gen-0"}
	al := NewAgentLoop(cfg, bus.NewMessageBus(), first)
	defer al.Close()

	// One reload first, so "live" means the current generation rather than the
	// zero value the loop starts on.
	provider := &scriptedProvider{name: "gen-1", replies: []scriptedReply{{text: "child answer"}}}
	if err := al.ReloadProviderAndConfig(
		context.Background(),
		provider,
		reloadTestConfig(t, workspace),
	); err != nil {
		t.Fatalf("reload: %v", err)
	}

	parent := &turnState{
		ctx:            context.Background(),
		turnID:         "live-parent",
		generation:     al.turns.currentGeneration(),
		childTurnIDs:   []string{},
		pendingResults: make(chan *tools.ToolResult, 4),
		session:        &ephemeralSessionStore{},
		agent:          al.GetRegistry().GetDefaultAgent(),
	}
	if _, err := spawnSubTurn(context.Background(), al, parent, SubTurnConfig{
		Model: "gpt-4o-mini",
		Tools: []tools.Tool{},
	}); err != nil {
		t.Fatalf("a sub-turn of a live parent was refused: %v", err)
	}
	if calls, _, _ := provider.stats(); calls == 0 {
		t.Error("the sub-turn never reached the provider")
	}
}

// A turn admitted after a reload records the new generation, so its children
// are admitted too.
func TestTurnsAdmittedAfterAReloadCarryTheNewGeneration(t *testing.T) {
	workspace := t.TempDir()
	cfg := reloadTestConfig(t, workspace)
	provider := &scriptedProvider{name: "gen-0", replies: []scriptedReply{{text: "answered"}}}

	al := NewAgentLoop(cfg, bus.NewMessageBus(), provider)
	defer al.Close()

	before := al.turns.currentGeneration()
	next := &scriptedProvider{name: "gen-1", replies: []scriptedReply{{text: "answered"}}}
	if err := al.ReloadProviderAndConfig(context.Background(), next, reloadTestConfig(t, workspace)); err != nil {
		t.Fatalf("reload: %v", err)
	}
	if after := al.turns.currentGeneration(); after == before {
		t.Fatalf("the generation did not advance across a commit (%d)", after)
	}

	if _, err := al.processMessage(context.Background(), inboundMessage("chat", "hello")); err != nil {
		t.Fatalf("turn after the reload: %v", err)
	}
	if _, _, usedAfterClose := provider.stats(); usedAfterClose {
		t.Error("the turn after the reload used the retired provider")
	}
}

// Media tools are rebuilt on every reload with a nil store. The process's store
// has to be handed to the candidate before it goes live, or a reload leaves the
// bot unable to read or send a file until someone restarts it.
func TestReloadKeepsTheInjectedMediaStore(t *testing.T) {
	workspace := t.TempDir()
	cfg := reloadTestConfig(t, workspace)
	cfg.Tools.LoadImage.Enabled = true
	cfg.Tools.SendFile.Enabled = true

	provider := &scriptedProvider{name: "gen-0", replies: []scriptedReply{{text: "hi"}}}
	al := NewAgentLoop(cfg, bus.NewMessageBus(), provider)
	defer al.Close()

	store := media.NewFileMediaStore()
	al.SetMediaStore(store)

	candidateCfg := reloadTestConfig(t, workspace)
	candidateCfg.Tools.LoadImage.Enabled = true
	candidateCfg.Tools.SendFile.Enabled = true
	if err := al.ReloadProviderAndConfig(
		context.Background(),
		&scriptedProvider{name: "gen-1", replies: []scriptedReply{{text: "hi"}}},
		candidateCfg,
	); err != nil {
		t.Fatalf("reload: %v", err)
	}

	if al.mediaStore != store {
		t.Error("the loop lost the injected media store")
	}
	// The tools expose no getter, so ask them to work: without a store they
	// answer "media store not configured" instead of failing on the path.
	image := filepath.Join(workspace, "not-an-image.txt")
	if err := os.WriteFile(image, []byte("plain text"), 0o644); err != nil {
		t.Fatal(err)
	}
	args := map[string]any{"path": image}
	for _, name := range []string{"load_image", "send_file"} {
		checked := 0
		al.GetRegistry().ForEachTool(name, func(tool tools.Tool) {
			checked++
			toolCtx := tools.WithToolContext(context.Background(), "cli", "chat")
			if result := tool.Execute(toolCtx, args); result != nil &&
				strings.Contains(result.ForLLM, "media store not configured") {
				t.Errorf("%s was rebuilt without the process media store", name)
			}
		})
		if checked == 0 {
			t.Errorf("%s is not registered after the reload", name)
		}
	}
}

// A refused reload must not disturb the store either.
func TestFailedReloadKeepsTheInjectedMediaStore(t *testing.T) {
	workspace := t.TempDir()
	cfg := reloadTestConfig(t, workspace)
	provider := &scriptedProvider{name: "gen-0", replies: []scriptedReply{{text: "hi"}}}

	al := NewAgentLoop(cfg, bus.NewMessageBus(), provider)
	defer al.Close()

	store := media.NewFileMediaStore()
	al.SetMediaStore(store)

	candidate := reloadTestConfig(t, workspace)
	candidate.Tools.MCP.Enabled = true
	candidate.Tools.MCP.Servers = map[string]config.MCPServerConfig{
		"broken": {Enabled: true, Command: filepath.Join(workspace, "no-such-mcp-binary")},
	}
	if err := al.ReloadProviderAndConfig(context.Background(), &scriptedProvider{name: "gen-1"}, candidate); err == nil {
		t.Fatal("the broken candidate was accepted")
	}
	if al.mediaStore != store {
		t.Error("a failed reload dropped the injected media store")
	}
}
