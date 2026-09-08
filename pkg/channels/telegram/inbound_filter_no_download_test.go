package telegram

import (
	"context"
	"strings"
	"sync"
	"testing"

	"github.com/mymmrac/telego"
	ta "github.com/mymmrac/telego/telegoapi"

	"github.com/sipeed/picoclaw/pkg/bus"
	"github.com/sipeed/picoclaw/pkg/channels"
)

// countingCaller records every Telegram API call the channel makes. getFile is
// the one that matters: without it there is no download URL, so no attachment
// can leave Telegram.
type countingCaller struct {
	mu    sync.Mutex
	calls []string
}

func (c *countingCaller) Call(_ context.Context, url string, _ *ta.RequestData) (*ta.Response, error) {
	c.mu.Lock()
	c.calls = append(c.calls, url)
	c.mu.Unlock()

	if strings.HasSuffix(url, "/getMe") {
		return &ta.Response{Ok: true, Result: []byte(`{"id":1,"is_bot":true,"first_name":"bot","username":"testbot"}`)}, nil
	}
	if strings.HasSuffix(url, "/getFile") {
		return &ta.Response{Ok: true, Result: []byte(`{"file_id":"f","file_unique_id":"u","file_path":"photos/file_1.jpg"}`)}, nil
	}
	return &ta.Response{Ok: true, Result: []byte("true")}, nil
}

func (c *countingCaller) countOf(suffix string) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	n := 0
	for _, url := range c.calls {
		if strings.HasSuffix(url, suffix) {
			n++
		}
	}
	return n
}

func newCountingChannel(t *testing.T, allowList ...string) (*TelegramChannel, *countingCaller, *bus.MessageBus) {
	t.Helper()
	caller := &countingCaller{}
	bot, err := telego.NewBot(
		"123456:"+strings.Repeat("a", 35),
		telego.WithAPICaller(caller),
		telego.WithDiscardLogger(),
	)
	if err != nil {
		t.Fatalf("NewBot: %v", err)
	}

	messageBus := bus.NewMessageBus()
	ch := &TelegramChannel{
		BaseChannel: channels.NewBaseChannel("telegram", nil, messageBus, allowList),
		bot:         bot,
		chatIDs:     make(map[string]int64),
		ctx:         context.Background(),
	}
	return ch, caller, messageBus
}

func photoMessage(id int, caption string) *telego.Message {
	return &telego.Message{
		MessageID: id,
		From:      &telego.User{ID: 42, Username: "sender", FirstName: "Sender"},
		Chat:      telego.Chat{ID: 99, Type: "private"},
		Caption:   caption,
		Photo: []telego.PhotoSize{{
			FileID: "photo-file-id", FileUniqueID: "u", Width: 100, Height: 100, FileSize: 1024,
		}},
	}
}

// A rejected message must cost nothing: no getFile, so no download URL, so no
// attachment ever leaves Telegram. This is the whole point of running the
// filter before anything is collected - a locked bot must not fetch what it was
// sent.
func TestRejectedInboundNeverAsksForAFile(t *testing.T) {
	cases := map[string][]*telego.Message{
		"single photo": {photoMessage(1, "")},
		"photo with caption": {
			photoMessage(1, "look at this"),
		},
		"album with caption": {
			photoMessage(1, "album caption"),
			photoMessage(2, ""),
			photoMessage(3, ""),
		},
		"quoted message": {func() *telego.Message {
			msg := photoMessage(1, "see the quote")
			msg.Quote = &telego.TextQuote{Text: "/auth hunter2"}
			return msg
		}()},
		"document": {func() *telego.Message {
			msg := photoMessage(1, "")
			msg.Photo = nil
			msg.Document = &telego.Document{FileID: "doc-file-id", FileName: "notes.txt", FileSize: 10}
			return msg
		}()},
	}

	for name, messages := range cases {
		t.Run(name, func(t *testing.T) {
			ch, caller, messageBus := newCountingChannel(t)

			rejected := 0
			ch.SetInboundFilter(func(_ context.Context, _ InboundRequest) (InboundDecision, error) {
				rejected++
				return InboundConsume, nil
			})

			if err := ch.handleMessages(context.Background(), messages); err != nil {
				t.Fatalf("handleMessages: %v", err)
			}

			if rejected != 1 {
				t.Errorf("the filter ran %d times, want exactly once for the batch", rejected)
			}
			if got := caller.countOf("/getFile"); got != 0 {
				t.Errorf("a rejected message triggered %d getFile call(s)", got)
			}
			select {
			case msg := <-messageBus.InboundChan():
				t.Errorf("a rejected message reached the bus: %q", msg.Content)
			default:
			}
		})
	}
}

// Control: with no filter installed the same album does ask for its files.
// Without this, the test above would pass on a channel that never downloads
// anything.
func TestAcceptedInboundDoesAskForItsFiles(t *testing.T) {
	ch, caller, _ := newCountingChannel(t)

	messages := []*telego.Message{
		photoMessage(1, "album caption"),
		photoMessage(2, ""),
	}
	if err := ch.handleMessages(context.Background(), messages); err != nil {
		t.Fatalf("handleMessages: %v", err)
	}

	if got := caller.countOf("/getFile"); got != len(messages) {
		t.Errorf("getFile calls = %d, want one per photo (%d)", got, len(messages))
	}
}

// The allowlist runs before the filter and must be just as cheap: a sender who
// is not on it costs no getFile either.
func TestUnallowedSenderNeverAsksForAFile(t *testing.T) {
	ch, caller, messageBus := newCountingChannel(t, "telegram:1")

	filterRan := false
	ch.SetInboundFilter(func(_ context.Context, _ InboundRequest) (InboundDecision, error) {
		filterRan = true
		return InboundPass, nil
	})

	messages := []*telego.Message{photoMessage(1, "album caption"), photoMessage(2, "")}
	if err := ch.handleMessages(context.Background(), messages); err != nil {
		t.Fatalf("handleMessages: %v", err)
	}

	if filterRan {
		t.Error("the filter ran for a sender the allowlist already rejected")
	}
	if got := caller.countOf("/getFile"); got != 0 {
		t.Errorf("an unallowed sender triggered %d getFile call(s)", got)
	}
	select {
	case msg := <-messageBus.InboundChan():
		t.Errorf("an unallowed message reached the bus: %q", msg.Content)
	default:
	}
}
