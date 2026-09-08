// PicoClaw - Ultra-lightweight personal AI agent
//
// Copyright (c) 2026 PicoClaw contributors

package agent

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

// ErrTurnsBusy reports that a generation change gave up waiting for turns that
// were already running. It is the one failure a caller can retry unchanged, so
// it is distinguishable rather than folded into a generic error.
var ErrTurnsBusy = errors.New("turns are still running")

// ErrSessionsNotPersistent reports that a generation is running on the
// in-memory session fallback, where a swap would drop the conversation instead
// of handing it over. It names a condition the operator can repair.
var ErrSessionsNotPersistent = errors.New("session store is not persistent")

// ErrStaleGeneration reports that the turn asking to spawn a sub-turn belongs
// to a generation that has already been retired. Its agent instance and
// provider are gone, so the child is refused instead of being handed them.
var ErrStaleGeneration = errors.New("the parent turn belongs to a retired generation")

// ErrReloadInProgress reports that another generation change owns the barrier.
// The caller must not touch the barrier when it sees this: it belongs to the
// change that is already running.
var ErrReloadInProgress = errors.New("another generation change is already in progress")

// turnBarrier lets a generation change happen at a point where no turn is
// running.
//
// A turn holds the barrier for its whole lifetime, not just for the LLM call it
// happens to be making. That distinction matters: a turn sitting in a tool call
// holds no LLM request, yet it will keep reading the loop's provider, hooks and
// MCP manager when the tool returns. Draining LLM requests alone would let a
// swap land in the middle of such a turn and hand it a mixture of two
// generations.
//
// block() closes admission first and only then waits, so a steady stream of new
// turns cannot starve a pending reload.
type turnBarrier struct {
	mu      sync.Mutex
	cond    *sync.Cond
	blocked bool
	active  int
	// holders are sub-turns. They never wait on the barrier - a child waiting
	// for the parent that holds it would deadlock - so they register instead,
	// and a generation change waits for them like any other work.
	holders int
	// generation counts commits. A turn records the value it started on, and a
	// sub-turn is admitted only if its parent still matches: counting what is
	// running cannot tell whether a particular parent's agent instance is still
	// the live one.
	generation uint64
}

func (b *turnBarrier) init() {
	b.cond = sync.NewCond(&b.mu)
}

// enter registers one turn. It waits while a generation change is in progress
// and reports the context error if the caller gives up first.
func (b *turnBarrier) enter(ctx context.Context) error {
	if b.cond == nil {
		return nil
	}

	b.mu.Lock()
	if b.blocked {
		stop := b.wakeOnDone(ctx)
		for b.blocked && ctx.Err() == nil {
			b.cond.Wait()
		}
		stop()
		if err := ctx.Err(); err != nil {
			b.mu.Unlock()
			return err
		}
	}
	b.active++
	b.mu.Unlock()
	return nil
}

// reserveSubTurn registers a sub-turn before it takes any reference to the
// running generation.
//
// Two things are checked, and both are needed. The parent's generation must
// still be the live one: an async spawn can be scheduled before a reload and
// reach here after it, and by then the parent's agent instance and provider
// have been retired - counting what is running says nothing about that
// particular parent. And a commit must not be in progress, which is exactly
// the state "admission closed, nothing left running": block() returns as soon
// as active and holders both reach zero, so while either is non-zero the old
// generation is guaranteed alive and the child may pin it.
//
// A refused child reports the error to its parent rather than waiting: waiting
// would only let it pin a generation that is on its way out.
func (b *turnBarrier) reserveSubTurn(parentGeneration uint64) error {
	if b.cond == nil {
		return nil
	}

	b.mu.Lock()
	defer b.mu.Unlock()
	if parentGeneration != b.generation {
		return ErrStaleGeneration
	}
	if b.blocked && b.active == 0 && b.holders == 0 {
		return ErrReloadInProgress
	}
	b.holders++
	return nil
}

// currentGeneration reports the generation a turn starting now belongs to.
func (b *turnBarrier) currentGeneration() uint64 {
	if b.cond == nil {
		return 0
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.generation
}

// advanceGeneration retires the current generation. It is called at the commit
// point, while admission is closed, so no turn can straddle the change.
func (b *turnBarrier) advanceGeneration() {
	if b.cond == nil {
		return
	}
	b.mu.Lock()
	b.generation++
	b.mu.Unlock()
}

// releaseSubTurn releases one sub-turn.
func (b *turnBarrier) releaseSubTurn() {
	if b.cond == nil {
		return
	}

	b.mu.Lock()
	if b.holders > 0 {
		b.holders--
	}
	if b.holders == 0 {
		b.cond.Broadcast()
	}
	b.mu.Unlock()
}

// subTurnCount reports how many sub-turns are registered.
func (b *turnBarrier) subTurnCount() int {
	if b.cond == nil {
		return 0
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.holders
}

// leave releases one turn.
func (b *turnBarrier) leave() {
	if b.cond == nil {
		return
	}

	b.mu.Lock()
	if b.active > 0 {
		b.active--
	}
	if b.active == 0 {
		b.cond.Broadcast()
	}
	b.mu.Unlock()
}

// block closes admission and waits until every turn already inside has
// finished. On any error admission is reopened, so a failed or abandoned
// generation change never leaves the loop refusing work.
func (b *turnBarrier) block(ctx context.Context, timeout time.Duration) error {
	if b.cond == nil {
		return nil
	}

	b.mu.Lock()
	if b.blocked {
		b.mu.Unlock()
		return ErrReloadInProgress
	}
	b.blocked = true
	b.cond.Broadcast()

	stop := b.wakeOnDone(ctx)
	var timedOut bool
	if timeout > 0 {
		timer := time.AfterFunc(timeout, func() {
			b.mu.Lock()
			timedOut = true
			b.cond.Broadcast()
			b.mu.Unlock()
		})
		defer timer.Stop()
	}

	// Sub-turns count too: a background one outlives the turn that spawned it
	// and would otherwise be the one holder nobody waits for.
	for (b.active > 0 || b.holders > 0) && !timedOut && ctx.Err() == nil {
		b.cond.Wait()
	}
	drained := b.active == 0 && b.holders == 0
	stop()

	if drained {
		b.mu.Unlock()
		return nil
	}

	b.blocked = false
	running := b.active + b.holders
	b.cond.Broadcast()
	b.mu.Unlock()

	if err := ctx.Err(); err != nil {
		return fmt.Errorf("canceled while waiting for %d running turn(s): %w", running, err)
	}
	return fmt.Errorf("%w: %d turn(s) after %s", ErrTurnsBusy, running, timeout)
}

// unblock reopens admission.
//
// Only the caller whose block() returned nil may call it. A caller that was
// refused never owned the barrier, and reopening it there would release the
// change that does own it - admitting turns into the middle of a commit.
func (b *turnBarrier) unblock() {
	if b.cond == nil {
		return
	}

	b.mu.Lock()
	b.blocked = false
	b.cond.Broadcast()
	b.mu.Unlock()
}

// wakeOnDone broadcasts once ctx is done, because sync.Cond cannot wait on a
// context. The returned function stops the watcher.
func (b *turnBarrier) wakeOnDone(ctx context.Context) func() {
	if ctx == nil || ctx.Done() == nil {
		return func() {}
	}

	done := make(chan struct{})
	go func() {
		select {
		case <-ctx.Done():
			b.mu.Lock()
			b.cond.Broadcast()
			b.mu.Unlock()
		case <-done:
		}
	}()
	return func() { close(done) }
}
