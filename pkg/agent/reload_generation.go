// PicoClaw - Ultra-lightweight personal AI agent
//
// Copyright (c) 2026 PicoClaw contributors

package agent

import (
	"context"
	"fmt"

	"github.com/sipeed/picoclaw/pkg/config"
	"github.com/sipeed/picoclaw/pkg/logger"
	"github.com/sipeed/picoclaw/pkg/mcp"
	"github.com/sipeed/picoclaw/pkg/providers"
	"github.com/sipeed/picoclaw/pkg/session"
)

// generation is everything a reload replaces at once: the config and the
// registry, plus the subsystems whose lifetime follows them.
//
// It exists so a candidate can be built and, if anything fails, thrown away
// with the running loop untouched. Nothing here is installed until commit.
type generation struct {
	cfg       *config.Config
	provider  providers.LLMProvider
	registry  *AgentRegistry
	evolution *evolutionBridge
	fallback  *providers.FallbackChain
	hooks     *HookManager
	mounted   []string
	mcp       *mcp.Manager
}

// prepareGeneration builds a complete candidate generation. Any failure returns
// an error with the candidate already cleaned up, so the caller can keep
// running on what it has.
//
// It must be called at a point where no turn is running: hooks and MCP register
// themselves against the registry being built, and the running loop must not be
// able to observe a half-built one.
func (al *AgentLoop) prepareGeneration(
	ctx context.Context,
	provider providers.LLMProvider,
	cfg *config.Config,
) (*generation, error) {
	gen := &generation{cfg: cfg, provider: provider}

	var registry *AgentRegistry
	func() {
		defer func() {
			if r := recover(); r != nil {
				logger.RecoverPanicNoExit(r)
				logger.ErrorCF("agent", "Panic during registry creation",
					map[string]any{"panic": r})
				registry = nil
			}
		}()
		registry = NewAgentRegistry(cfg, provider)
	}()
	if registry == nil {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("context canceled during registry creation: %w", err)
		}
		return nil, fmt.Errorf("registry creation failed")
	}
	gen.registry = registry

	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context canceled after registry creation: %w", err)
	}

	registerSharedTools(al, cfg, al.bus, registry, provider)

	// Shared dependencies injected into the loop are not config: they belong to
	// the process and outlive every generation. The media store is one, and the
	// tools that need it are rebuilt nil above, so it is handed over here -
	// before the commit, not after, or the live registry would briefly carry
	// media tools that cannot do anything.
	propagateMediaStore(registry, al.mediaStore)

	// The evolution bridge stays a warning: it is optional at startup
	// (agent_init.go) and this patch does not change which failures are fatal
	// beyond the ones it is about.
	evolution, evolutionErr := newEvolutionBridge(registry, cfg, provider)
	if evolutionErr != nil {
		logger.WarnCF("agent", "Failed to reinitialize evolution bridge during reload",
			map[string]any{"error": evolutionErr.Error()})
	}
	gen.evolution = evolution

	rl := providers.NewRateLimiterRegistry()
	for _, agentID := range registry.ListAgentIDs() {
		if agent, ok := registry.GetAgent(agentID); ok {
			rl.RegisterCandidates(agent.Candidates)
			rl.RegisterCandidates(agent.LightCandidates)
		}
	}
	gen.fallback = providers.NewFallbackChain(providers.NewCooldownTracker(), rl)

	gen.hooks = NewHookManager(al.runtimeEvents.Channel())
	configureHookManagerFromConfig(gen.hooks, cfg)
	mounted, hookErr := mountConfiguredHooks(ctx, gen.hooks, cfg)
	if hookErr != nil {
		gen.close()
		return nil, fmt.Errorf("hooks: %w", hookErr)
	}
	gen.mounted = mounted

	if mcpCfg, start := mcpServersToStart(cfg, registry); start {
		manager, mcpErr := startMCPServers(ctx, cfg, registry, al.runtimeEvents, mcpCfg)
		if mcpErr != nil {
			gen.close()
			return nil, fmt.Errorf("mcp: %w", mcpErr)
		}
		gen.mcp = manager
	}

	// A generation on the in-memory fallback has nothing on disk to hand over,
	// so committing would drop the conversation while reporting success. Refuse
	// and say what to repair.
	if store, ok := nonPersistentSessionStore(al.registry); ok {
		gen.close()
		return nil, fmt.Errorf(
			"%w (running generation uses %T): fix the sessions directory, then reload",
			ErrSessionsNotPersistent, store,
		)
	}
	if store, ok := nonPersistentSessionStore(registry); ok {
		gen.close()
		return nil, fmt.Errorf(
			"%w (candidate uses %T): fix the sessions directory, then reload",
			ErrSessionsNotPersistent, store,
		)
	}

	if err := ctx.Err(); err != nil {
		gen.close()
		return nil, fmt.Errorf("context canceled after preparing the new generation: %w", err)
	}

	return gen, nil
}

// nonPersistentSessionStore reports an agent whose sessions live only in
// memory. instance.go falls back to that store when the on-disk one cannot be
// created, which means history is already being lost at every restart.
func nonPersistentSessionStore(registry *AgentRegistry) (session.SessionStore, bool) {
	if registry == nil {
		return nil, false
	}
	for _, agentID := range registry.ListAgentIDs() {
		agent, ok := registry.GetAgent(agentID)
		if !ok || agent == nil || agent.Sessions == nil {
			continue
		}
		if _, isManager := agent.Sessions.(*session.SessionManager); isManager {
			return agent.Sessions, true
		}
	}
	return nil, false
}

// commit installs the candidate and retires what it replaces. It must be called
// while the turn barrier is held, so no turn can read one half of the swap.
func (al *AgentLoop) commit(gen *generation) {
	// Hooks mounted through MountHook belong to whoever mounted them, not to
	// the configuration being replaced. Carry them onto the new manager before
	// the old one is closed, or a reload silently unregisters them.
	fromConfig := make(map[string]struct{}, len(gen.mounted))
	for _, name := range gen.mounted {
		fromConfig[name] = struct{}{}
	}
	for _, reg := range al.hooks.takeUnmanaged(al.hookRuntime.mountedNames()) {
		if _, taken := fromConfig[reg.Name]; taken {
			// The new configuration mounts a hook under this name. Leave the
			// external one detached rather than closed: its owner still holds
			// it and can mount it again.
			logger.WarnCF("agent", "An externally mounted hook was dropped: the new config uses its name",
				map[string]any{"hook": reg.Name})
			continue
		}
		if err := gen.hooks.Mount(reg); err != nil {
			logger.WarnCF("agent", "Failed to carry an externally mounted hook across the reload",
				map[string]any{"hook": reg.Name, "error": err.Error()})
		}
	}

	al.mu.Lock()
	oldRegistry := al.registry
	oldEvolution := al.evolution
	oldHooks := al.hooks

	al.cfg = gen.cfg
	al.registry = gen.registry
	al.evolution = gen.evolution
	al.fallback = gen.fallback
	al.hooks = gen.hooks
	al.mu.Unlock()

	if gen.evolution != nil {
		gen.evolution.setCurrentCheck(al.isCurrentEvolutionBridge)
		if err := gen.evolution.subscribeRuntimeEvents(al.runtimeEvents.Channel()); err != nil {
			logger.WarnCF("agent", "Failed to subscribe reloaded evolution bridge to runtime events",
				map[string]any{"error": err.Error()})
		}
	}

	// From here the previous generation is retired: a sub-turn whose parent
	// started on it is refused rather than handed a closed provider.
	al.turns.advanceGeneration()

	al.hookRuntime.adopt(gen.mounted)
	oldMCPManager := al.mcp.adopt(gen.mcp)

	al.refreshRuntimeEventLogger(gen.cfg)

	// Everything below belongs to the generation that just retired. The barrier
	// guarantees no turn is still using it, so each is closed exactly once here
	// rather than left for the garbage collector or a timer.
	if oldMCPManager != nil {
		if err := oldMCPManager.Close(); err != nil {
			logger.WarnCF("agent", "Failed to close previous MCP manager during reload",
				map[string]any{"error": err.Error()})
		}
	}
	if oldEvolution != nil {
		if err := oldEvolution.Close(); err != nil {
			logger.WarnCF("agent", "Failed to close previous evolution bridge during reload",
				map[string]any{"error": err.Error()})
		}
	}
	if oldHooks != nil && oldHooks != gen.hooks {
		oldHooks.Close()
	}
	if oldProvider, ok := extractProvider(oldRegistry); ok {
		if stateful, ok := oldProvider.(providers.StatefulProvider); ok {
			stateful.Close()
		}
	}
}

// close releases everything a candidate managed to build. It is only ever
// called on a candidate that was never committed.
func (gen *generation) close() {
	if gen == nil {
		return
	}
	if gen.mcp != nil {
		if err := gen.mcp.Close(); err != nil {
			logger.WarnCF("agent", "Failed to close candidate MCP manager",
				map[string]any{"error": err.Error()})
		}
		gen.mcp = nil
	}
	if gen.hooks != nil {
		for _, name := range gen.mounted {
			gen.hooks.Unmount(name)
		}
		gen.hooks.Close()
		gen.hooks = nil
		gen.mounted = nil
	}
	if gen.evolution != nil {
		if err := gen.evolution.Close(); err != nil {
			logger.WarnCF("agent", "Failed to close candidate evolution bridge",
				map[string]any{"error": err.Error()})
		}
		gen.evolution = nil
	}
	// The candidate registry and its session stores are dropped, not closed:
	// closing a registry closes session stores the retiring generation may
	// still own, and the candidate never owned a provider of its own.
	gen.registry = nil
}
