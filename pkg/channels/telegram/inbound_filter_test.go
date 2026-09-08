package telegram

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mymmrac/telego"
	ta "github.com/mymmrac/telego/telegoapi"

	"github.com/sipeed/picoclaw/pkg/bus"
	"github.com/sipeed/picoclaw/pkg/channels"
	"github.com/sipeed/picoclaw/pkg/commands"
	"github.com/sipeed/picoclaw/pkg/config"
)

// newFilterTestChannel builds a channel wired to a real message bus and a
// recording API caller, so tests can observe both publication and transport
// calls.
func newFilterTestChannel(t *testing.T) (*TelegramChannel, *bus.MessageBus, *stubCaller) {
	t.Helper()

	caller := &stubCaller{
		callFn: func(_ context.Context, url string, _ *ta.RequestData) (*ta.Response, error) {
			if strings.HasSuffix(url, "/getMe") {
				return &ta.Response{
					Ok:     true,
					Result: []byte(`{"id":1,"is_bot":true,"first_name":"bot","username":"testbot"}`),
				}, nil
			}
			msg, err := json.Marshal(&telego.Message{MessageID: 999})
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			return &ta.Response{Ok: true, Result: msg}, nil
		},
	}

	bot, err := telego.NewBot(testToken,
		telego.WithAPICaller(caller),
		telego.WithRequestConstructor(&stubConstructor{}),
		telego.WithDiscardLogger(),
	)
	if err != nil {
		t.Fatalf("NewBot: %v", err)
	}

	messageBus := bus.NewMessageBus()
	ch := &TelegramChannel{
		BaseChannel: channels.NewBaseChannel("telegram", nil, messageBus, nil),
		bot:         bot,
		bc:          &config.Channel{Type: config.ChannelTelegram, Enabled: true},
		tgCfg:       &config.TelegramSettings{},
		chatIDs:     make(map[string]int64),
		ctx:         context.Background(),
	}
	return ch, messageBus, caller
}

func textMessage(id int, text string) *telego.Message {
	return &telego.Message{
		MessageID: id,
		Text:      text,
		Chat:      telego.Chat{ID: 123, Type: "private"},
		From:      &telego.User{ID: 7, FirstName: "Alice", Username: "alice"},
	}
}

// expectPublished reports whether an inbound message reached the bus.
func expectPublished(t *testing.T, messageBus *bus.MessageBus) (bus.InboundMessage, bool) {
	t.Helper()
	select {
	case inbound, ok := <-messageBus.InboundChan():
		return inbound, ok
	case <-time.After(150 * time.Millisecond):
		return bus.InboundMessage{}, false
	}
}

func TestInboundFilter_NilFilterKeepsUpstreamBehaviour(t *testing.T) {
	ch, messageBus, _ := newFilterTestChannel(t)

	if err := ch.handleMessage(context.Background(), textMessage(42, "hello")); err != nil {
		t.Fatalf("handleMessage: %v", err)
	}

	inbound, ok := expectPublished(t, messageBus)
	if !ok {
		t.Fatal("expected the message to be published when no filter is installed")
	}
	if inbound.Content != "hello" {
		t.Fatalf("content = %q", inbound.Content)
	}
}

func TestInboundFilter_PassPublishesExactlyOnce(t *testing.T) {
	ch, messageBus, _ := newFilterTestChannel(t)

	var calls atomic.Int32
	ch.SetInboundFilter(func(context.Context, InboundRequest) (InboundDecision, error) {
		calls.Add(1)
		return InboundPass, nil
	})

	if err := ch.handleMessage(context.Background(), textMessage(42, "hello")); err != nil {
		t.Fatalf("handleMessage: %v", err)
	}
	if _, ok := expectPublished(t, messageBus); !ok {
		t.Fatal("expected the message to be published on InboundPass")
	}
	if _, ok := expectPublished(t, messageBus); ok {
		t.Fatal("expected exactly one publication")
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("filter called %d times, want 1", got)
	}
}

func TestInboundFilter_StopsPipelineBeforeAnySideEffect(t *testing.T) {
	cases := []struct {
		name     string
		decision InboundDecision
		err      error
	}{
		{name: "consume", decision: InboundConsume},
		{name: "reject", decision: InboundReject},
		{name: "error fails closed", decision: InboundPass, err: errors.New("secret-bearing failure: hunter2")},
		{name: "unknown decision fails closed", decision: InboundDecision(200)},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ch, messageBus, caller := newFilterTestChannel(t)
			ch.SetInboundFilter(func(context.Context, InboundRequest) (InboundDecision, error) {
				return tc.decision, tc.err
			})

			msg := textMessage(42, "hello")
			msg.Photo = []telego.PhotoSize{{FileID: "file-1", FileSize: 10}}

			if err := ch.handleMessage(context.Background(), msg); err != nil {
				t.Fatalf("handleMessage returned an error, which would make Telegram retry: %v", err)
			}
			if _, ok := expectPublished(t, messageBus); ok {
				t.Fatal("message reached the bus")
			}
			if len(ch.chatIDs) != 0 {
				t.Fatalf("chat tracking was mutated: %v", ch.chatIDs)
			}
			for _, call := range caller.calls {
				if strings.Contains(call.URL, "getFile") {
					t.Fatalf("media download was attempted: %s", call.URL)
				}
			}
		})
	}
}

func TestInboundRequest_Metadata(t *testing.T) {
	cases := []struct {
		name     string
		message  *telego.Message
		wantChat string
		wantType string
		wantThrd string
		wantText string
	}{
		{
			name:     "private text",
			message:  textMessage(42, "hello"),
			wantChat: "123",
			wantType: "private",
			wantText: "hello",
		},
		{
			name: "forum topic",
			message: &telego.Message{
				MessageID:       7,
				Text:            "in topic",
				MessageThreadID: 55,
				Chat:            telego.Chat{ID: -100, Type: "supergroup", IsForum: true},
				From:            &telego.User{ID: 7},
			},
			wantChat: "-100",
			wantType: "supergroup",
			wantThrd: "55",
			wantText: "in topic",
		},
		{
			name: "caption falls back",
			message: &telego.Message{
				MessageID: 8,
				Caption:   "a caption",
				Photo:     []telego.PhotoSize{{FileID: "f"}},
				Chat:      telego.Chat{ID: 123, Type: "private"},
				From:      &telego.User{ID: 7},
			},
			wantChat: "123",
			wantType: "private",
			wantText: "a caption",
		},
		{
			name: "media only has empty text but still reaches the filter",
			message: &telego.Message{
				MessageID: 9,
				Photo:     []telego.PhotoSize{{FileID: "f"}},
				Chat:      telego.Chat{ID: 123, Type: "private"},
				From:      &telego.User{ID: 7},
			},
			wantChat: "123",
			wantType: "private",
			wantText: "",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ch, _, _ := newFilterTestChannel(t)

			var got InboundRequest
			var called bool
			ch.SetInboundFilter(func(_ context.Context, req InboundRequest) (InboundDecision, error) {
				got, called = req, true
				return InboundReject, nil
			})

			if err := ch.handleMessage(context.Background(), tc.message); err != nil {
				t.Fatalf("handleMessage: %v", err)
			}
			if !called {
				t.Fatal("filter was not called")
			}
			if got.ChatID != tc.wantChat || got.ChatType != tc.wantType ||
				got.ThreadID != tc.wantThrd || got.Text != tc.wantText {
				t.Fatalf("request = %+v", got)
			}
			if got.Sender.Platform != "telegram" || got.Sender.PlatformID != "7" {
				t.Fatalf("sender = %+v", got.Sender)
			}
			if len(got.Messages) != 1 || got.Messages[0] != tc.message {
				t.Fatalf("Messages = %v, want the single raw message", got.Messages)
			}
		})
	}
}

func TestInboundRequest_ReplyAndDeleteTargetTheRightMessage(t *testing.T) {
	ch, _, caller := newFilterTestChannel(t)

	msg := &telego.Message{
		MessageID:       77,
		Text:            "hello",
		MessageThreadID: 55,
		Chat:            telego.Chat{ID: -100, Type: "supergroup", IsForum: true},
		From:            &telego.User{ID: 7},
	}

	ch.SetInboundFilter(func(ctx context.Context, req InboundRequest) (InboundDecision, error) {
		if err := req.Reply(ctx, "locked"); err != nil {
			t.Errorf("Reply: %v", err)
		}
		if err := req.Delete(ctx); err != nil {
			t.Errorf("Delete: %v", err)
		}
		return InboundConsume, nil
	})

	if err := ch.handleMessage(context.Background(), msg); err != nil {
		t.Fatalf("handleMessage: %v", err)
	}

	var sendBody, deleteBody string
	for _, call := range caller.calls {
		switch {
		case strings.HasSuffix(call.URL, "/sendMessage"):
			sendBody = string(call.Data.BodyRaw)
		case strings.HasSuffix(call.URL, "/deleteMessage"):
			deleteBody = string(call.Data.BodyRaw)
		}
	}
	if sendBody == "" || deleteBody == "" {
		t.Fatalf("missing transport calls: send=%q delete=%q", sendBody, deleteBody)
	}
	for _, want := range []string{`"chat_id":-100`, `"message_thread_id":55`, `"text":"locked"`} {
		if !strings.Contains(sendBody, want) {
			t.Errorf("sendMessage body %s missing %s", sendBody, want)
		}
	}
	for _, want := range []string{`"chat_id":-100`, `"message_id":77`} {
		if !strings.Contains(deleteBody, want) {
			t.Errorf("deleteMessage body %s missing %s", deleteBody, want)
		}
	}
}

func TestInboundRequest_HelpersRejectUnboundCopies(t *testing.T) {
	var escaped InboundRequest
	ch, _, _ := newFilterTestChannel(t)
	ch.SetInboundFilter(func(_ context.Context, req InboundRequest) (InboundDecision, error) {
		escaped = InboundRequest{Sender: req.Sender, ChatID: req.ChatID}
		return InboundReject, nil
	})
	if err := ch.handleMessage(context.Background(), textMessage(42, "hello")); err != nil {
		t.Fatalf("handleMessage: %v", err)
	}
	if err := escaped.Reply(context.Background(), "late"); !errors.Is(err, errInboundReplyUnavailable) {
		t.Fatalf("Reply on an unbound copy returned %v", err)
	}
	if err := escaped.Delete(context.Background()); !errors.Is(err, errInboundReplyUnavailable) {
		t.Fatalf("Delete on an unbound copy returned %v", err)
	}
}

func TestInboundFilter_AlbumIsOneCallWithTheWholeBatch(t *testing.T) {
	ch, _, _ := newFilterTestChannel(t)

	quoted := &telego.Message{
		MessageID: 1,
		Text:      "quoted secret",
		Chat:      telego.Chat{ID: 123, Type: "private"},
		From:      &telego.User{ID: 9},
	}
	first := &telego.Message{
		MessageID:      10,
		Photo:          []telego.PhotoSize{{FileID: "a"}},
		Chat:           telego.Chat{ID: 123, Type: "private"},
		From:           &telego.User{ID: 7},
		MediaGroupID:   "album-1",
		ReplyToMessage: quoted,
	}
	second := &telego.Message{
		MessageID:    11,
		Caption:      "second caption",
		Photo:        []telego.PhotoSize{{FileID: "b"}},
		Chat:         telego.Chat{ID: 123, Type: "private"},
		From:         &telego.User{ID: 7},
		MediaGroupID: "album-1",
	}

	var calls int
	var got InboundRequest
	ch.SetInboundFilter(func(_ context.Context, req InboundRequest) (InboundDecision, error) {
		calls++
		got = req
		return InboundReject, nil
	})

	if err := ch.handleMessages(context.Background(), []*telego.Message{first, second}); err != nil {
		t.Fatalf("handleMessages: %v", err)
	}
	if calls != 1 {
		t.Fatalf("filter called %d times for one album", calls)
	}
	if len(got.Messages) != 2 {
		t.Fatalf("batch has %d members, want 2", len(got.Messages))
	}
	// The representative is the first member carrying text or caption, so the
	// non-representative caption is only reachable through Messages.
	if got.Text != "second caption" || got.MessageID != "11" {
		t.Fatalf("representative = %q/%q", got.Text, got.MessageID)
	}
	if got.Messages[0].ReplyToMessage == nil || got.Messages[0].ReplyToMessage.Text != "quoted secret" {
		t.Fatal("quoted message was stripped before the filter")
	}
}

func TestInboundFilter_InconsistentBatchFailsClosedWithoutCallingBack(t *testing.T) {
	base := func() *telego.Message {
		return &telego.Message{
			MessageID:    10,
			Caption:      "one",
			Chat:         telego.Chat{ID: 123, Type: "private"},
			From:         &telego.User{ID: 7},
			MediaGroupID: "album-1",
		}
	}

	cases := map[string]*telego.Message{
		"nil member": nil,
		"missing sender": {
			MessageID: 11, Caption: "two",
			Chat: telego.Chat{ID: 123, Type: "private"},
		},
		"other sender": {
			MessageID: 11, Caption: "two",
			Chat: telego.Chat{ID: 123, Type: "private"},
			From: &telego.User{ID: 8},
		},
		"other chat": {
			MessageID: 11, Caption: "two",
			Chat: telego.Chat{ID: 456, Type: "private"},
			From: &telego.User{ID: 7},
		},
		"other topic": {
			MessageID: 11, Caption: "two", MessageThreadID: 3,
			Chat: telego.Chat{ID: 123, Type: "private"},
			From: &telego.User{ID: 7},
		},
	}

	for name, bad := range cases {
		t.Run(name, func(t *testing.T) {
			ch, messageBus, _ := newFilterTestChannel(t)
			var called bool
			ch.SetInboundFilter(func(context.Context, InboundRequest) (InboundDecision, error) {
				called = true
				return InboundPass, nil
			})

			if err := ch.handleMessages(context.Background(), []*telego.Message{base(), bad}); err != nil {
				t.Fatalf("handleMessages: %v", err)
			}
			if called {
				t.Fatal("filter was called for an inconsistent batch")
			}
			if _, ok := expectPublished(t, messageBus); ok {
				t.Fatal("inconsistent batch reached the bus")
			}
		})
	}
}

func TestInboundFilter_InconsistentBatchStillPassesWithoutFilter(t *testing.T) {
	ch, messageBus, _ := newFilterTestChannel(t)

	first := textMessage(10, "hello")
	second := textMessage(11, "world")
	second.From = &telego.User{ID: 8}

	if err := ch.handleMessages(context.Background(), []*telego.Message{first, second}); err != nil {
		t.Fatalf("handleMessages: %v", err)
	}
	if _, ok := expectPublished(t, messageBus); !ok {
		t.Fatal("nil-filter behaviour changed for mixed batches")
	}
}

func TestInboundFilter_ConcurrentSetAndInvoke(t *testing.T) {
	ch, _, _ := newFilterTestChannel(t)

	var wg sync.WaitGroup
	stop := make(chan struct{})

	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; ; i++ {
			select {
			case <-stop:
				return
			default:
			}
			if i%2 == 0 {
				ch.SetInboundFilter(func(context.Context, InboundRequest) (InboundDecision, error) {
					return InboundReject, nil
				})
			} else {
				ch.SetInboundFilter(nil)
			}
		}
	}()

	for i := range 50 {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			_ = ch.handleMessage(context.Background(), textMessage(id, "hello"))
		}(i)
	}

	time.Sleep(20 * time.Millisecond)
	close(stop)
	wg.Wait()
}

func TestCommandDefinitions_DefaultsToBuiltins(t *testing.T) {
	ch, _, _ := newFilterTestChannel(t)

	got := ch.commandDefinitions()
	want := commands.BuiltinDefinitions()
	if len(got) != len(want) {
		t.Fatalf("got %d definitions, want the %d builtins", len(got), len(want))
	}
	for i := range got {
		if got[i].Name != want[i].Name {
			t.Fatalf("definition %d = %q, want %q", i, got[i].Name, want[i].Name)
		}
	}
}

func TestCommandDefinitions_CustomSnapshotIsIndependent(t *testing.T) {
	ch, _, _ := newFilterTestChannel(t)

	custom := []commands.Definition{
		{Name: "auth", Description: "Unlock"},
		{Name: "unauth", Description: "Lock"},
	}
	ch.SetCommandDefinitions(custom)

	custom[0].Name = "mutated"
	custom = append(custom, commands.Definition{Name: "extra"})

	got := ch.commandDefinitions()
	if len(got) != 2 || got[0].Name != "auth" || got[1].Name != "unauth" {
		t.Fatalf("definitions = %+v, want an independent snapshot", got)
	}

	ch.SetCommandDefinitions(nil)
	if len(ch.commandDefinitions()) != len(commands.BuiltinDefinitions()) {
		t.Fatal("nil did not restore the builtin definitions")
	}
}

func TestConcurrentCommandDefinitionsAndFilter(t *testing.T) {
	ch, _, _ := newFilterTestChannel(t)

	var wg sync.WaitGroup
	for i := range 20 {
		wg.Add(2)
		go func(id int) {
			defer wg.Done()
			ch.SetCommandDefinitions([]commands.Definition{{Name: "auth", Description: "Unlock"}})
		}(i)
		go func() {
			defer wg.Done()
			_ = ch.commandDefinitions()
		}()
	}
	wg.Wait()
}
