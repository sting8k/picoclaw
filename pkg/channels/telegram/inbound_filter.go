// PicoClaw - Ultra-lightweight personal AI agent
// License: MIT
//
// Copyright (c) 2026 PicoClaw contributors

package telegram

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/mymmrac/telego"
	tu "github.com/mymmrac/telego/telegoutil"

	"github.com/sipeed/picoclaw/pkg/bus"
	"github.com/sipeed/picoclaw/pkg/commands"
	"github.com/sipeed/picoclaw/pkg/logger"
)

// InboundDecision tells the channel what to do with an inbound message after
// an InboundFilter has inspected it.
type InboundDecision uint8

const (
	// InboundPass continues the normal Telegram pipeline.
	InboundPass InboundDecision = iota
	// InboundConsume means the filter already handled the message (for example
	// by replying to it); the channel stops and publishes nothing.
	InboundConsume
	// InboundReject drops the message silently; the channel stops and
	// publishes nothing.
	InboundReject
)

// errInboundReplyUnavailable is returned by the request helpers on a request
// the channel did not build, such as a zero value or a partial copy.
var errInboundReplyUnavailable = errors.New("telegram: inbound request has no transport bound to it")

// InboundRequest is the read-only view of an inbound message handed to an
// InboundFilter.
//
// Messages is the full batch the channel is about to process: a media group in
// pipeline order, or a single-element slice for a normal message. The other
// fields describe the representative message the channel picked, so filters
// that only care about one message do not have to walk the batch.
//
// Messages and everything reachable from it is borrowed for the duration of the
// callback: do not mutate or retain it. Reply and Delete stay usable
// afterwards, but retaining the request is not part of the contract.
type InboundRequest struct {
	Sender    bus.SenderInfo
	ChatID    string            // raw numeric chat ID, without the forum topic suffix
	ChatType  string            // Telegram chat type: private, group, supergroup, channel
	ThreadID  string            // empty when the message is not in a forum topic
	MessageID string            // representative message ID
	Text      string            // representative Text, or Caption when Text is empty
	Messages  []*telego.Message // whole raw batch, including quoted and replied messages

	reply  func(context.Context, string) error
	delete func(context.Context) error
}

// Reply sends plain text straight to the originating chat and topic using the
// bot transport. It does not go through the message bus, and creates no
// placeholder or session, so it works before any agent runtime exists.
func (r InboundRequest) Reply(ctx context.Context, text string) error {
	if r.reply == nil {
		return errInboundReplyUnavailable
	}
	return r.reply(ctx, text)
}

// Delete removes the representative inbound message. It is best-effort: for a
// media group only the representative message is deleted, and quoted or replied
// messages are never touched.
func (r InboundRequest) Delete(ctx context.Context) error {
	if r.delete == nil {
		return errInboundReplyUnavailable
	}
	return r.delete(ctx)
}

// InboundFilter inspects an inbound message before the channel touches it.
//
// It is called synchronously, after the allowlist check and before chat
// tracking, media download, group and quote processing, debug previews and
// message publication. A nil filter is a no-op equivalent to InboundPass.
//
// Returning an error or an unknown decision fails closed: the message is
// dropped and nothing about its content is logged.
type InboundFilter func(context.Context, InboundRequest) (InboundDecision, error)

// SetInboundFilter installs the inbound filter. Passing nil removes it and
// restores the default pipeline. Safe to call concurrently with message
// handling.
func (c *TelegramChannel) SetInboundFilter(filter InboundFilter) {
	c.inboundMu.Lock()
	defer c.inboundMu.Unlock()
	c.inboundFilter = filter
}

// inboundFilterFn returns the installed filter, or nil.
func (c *TelegramChannel) inboundFilterFn() InboundFilter {
	c.inboundMu.RLock()
	defer c.inboundMu.RUnlock()
	return c.inboundFilter
}

// SetCommandDefinitions overrides the command definitions registered as the
// Telegram command menu on Start. Passing nil restores the built-in
// definitions. The slice is copied, so later changes by the caller are ignored.
func (c *TelegramChannel) SetCommandDefinitions(defs []commands.Definition) {
	c.inboundMu.Lock()
	defer c.inboundMu.Unlock()
	if defs == nil {
		c.commandDefs = nil
		return
	}
	c.commandDefs = slices.Clone(defs)
}

// commandDefinitions returns the definitions to register, defaulting to the
// built-in set when no override was installed.
func (c *TelegramChannel) commandDefinitions() []commands.Definition {
	c.inboundMu.RLock()
	defer c.inboundMu.RUnlock()
	if c.commandDefs == nil {
		return commands.BuiltinDefinitions()
	}
	return slices.Clone(c.commandDefs)
}

// applyInboundFilter runs the installed filter for a batch and reports whether
// the channel should keep processing it.
//
// When a filter is installed, a batch whose members disagree about sender, chat
// or topic is rejected before the callback runs: such a batch cannot be
// attributed to one authorization decision.
func (c *TelegramChannel) applyInboundFilter(
	ctx context.Context,
	messages []*telego.Message,
	message *telego.Message,
	sender bus.SenderInfo,
) bool {
	filter := c.inboundFilterFn()
	if filter == nil {
		return true
	}

	logFields := map[string]any{
		"sender_id":  sender.CanonicalID,
		"chat_id":    message.Chat.ID,
		"message_id": message.MessageID,
		"batch_size": len(messages),
	}

	if !batchIsConsistent(messages, message) {
		logger.WarnCF("telegram", "Inbound batch rejected: members disagree on sender, chat or topic", logFields)
		return false
	}

	decision, err := filter(ctx, c.newInboundRequest(messages, message, sender))
	if err != nil {
		// The error may quote message content, so only fixed text and metadata
		// are logged, and the error is not returned to the handler: Telegram
		// would retry the update and replay the content.
		logger.ErrorCF("telegram", "Inbound filter failed; message dropped", logFields)
		return false
	}

	switch decision {
	case InboundPass:
		return true
	case InboundConsume, InboundReject:
		return false
	default:
		logger.ErrorCF("telegram", "Inbound filter returned an unknown decision; message dropped", logFields)
		return false
	}
}

// newInboundRequest builds the read-only view passed to the filter. The reply
// and delete helpers are bound to the representative message. They keep
// working after the callback returns, but callers must not retain them: the
// batch they describe is only valid for the duration of the call.
func (c *TelegramChannel) newInboundRequest(
	messages []*telego.Message,
	message *telego.Message,
	sender bus.SenderInfo,
) InboundRequest {
	chatID := message.Chat.ID
	threadID := message.MessageThreadID

	req := InboundRequest{
		Sender:    sender,
		ChatID:    fmt.Sprintf("%d", chatID),
		ChatType:  message.Chat.Type,
		MessageID: fmt.Sprintf("%d", message.MessageID),
		Text:      representativeText(message),
		Messages:  messages,
	}
	if threadID != 0 {
		req.ThreadID = fmt.Sprintf("%d", threadID)
	}

	req.reply = func(ctx context.Context, text string) error {
		if strings.TrimSpace(text) == "" {
			return fmt.Errorf("telegram: inbound reply text is empty")
		}
		params := tu.Message(tu.ID(chatID), text)
		params.MessageThreadID = threadID
		_, err := c.bot.SendMessage(ctx, params)
		return err
	}
	req.delete = func(ctx context.Context) error {
		return c.bot.DeleteMessage(ctx, &telego.DeleteMessageParams{
			ChatID:    tu.ID(chatID),
			MessageID: message.MessageID,
		})
	}
	return req
}

// representativeText mirrors how the pipeline picks the text of a message:
// Text when present, otherwise Caption.
func representativeText(message *telego.Message) string {
	if message.Text != "" {
		return message.Text
	}
	return message.Caption
}

// batchIsConsistent reports whether every member of the batch is a usable
// message from the same sender, chat and topic as the representative.
func batchIsConsistent(messages []*telego.Message, message *telego.Message) bool {
	if message == nil || message.From == nil {
		return false
	}
	if len(messages) == 0 {
		return false
	}
	for _, candidate := range messages {
		if candidate == nil || candidate.From == nil {
			return false
		}
		if candidate.From.ID != message.From.ID {
			return false
		}
		if candidate.Chat.ID != message.Chat.ID {
			return false
		}
		if candidate.MessageThreadID != message.MessageThreadID {
			return false
		}
	}
	return true
}
