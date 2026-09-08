package telegram

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/mymmrac/telego"

	"github.com/sipeed/picoclaw/pkg/logger"
)

// canaryToken is a well-formed but invented bot token. A real one must never
// appear in a test, a repro or a report.
var canaryToken = "123456789:AAFcanary" + strings.Repeat("z", 26) // 35 chars after the colon

// The download URL is built from the bot token, and it used to be written to
// the debug log verbatim. Anyone with that line can act as the bot.
func TestFileDownloadURLIsNotLoggedWithTheToken(t *testing.T) {
	bot, err := telego.NewBot(canaryToken, telego.WithDiscardLogger())
	if err != nil {
		t.Fatalf("NewBot: %v", err)
	}
	channel := &TelegramChannel{bot: bot}

	path := filepath.Join(t.TempDir(), "picoclaw.log")
	if err := logger.EnableFileLogging(path); err != nil {
		t.Fatalf("EnableFileLogging: %v", err)
	}
	previous := logger.GetLevel()
	logger.SetLevel(logger.DEBUG)

	// The download itself fails without network; the debug line is written
	// before that, which is the line under test.
	channel.downloadFileWithInfo(&telego.File{FilePath: "photos/file_1.jpg"}, ".jpg")

	logger.SetLevel(previous)
	logger.DisableFileLogging()

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read log: %v", err)
	}
	logs := string(data)

	if strings.Contains(logs, canaryToken) {
		t.Errorf("the bot token reached the log:\n%s", logs)
	}
	if !strings.Contains(logs, "bot<redacted>") {
		t.Errorf("the redacted URL was not logged at all:\n%s", logs)
	}
	if !strings.Contains(logs, "photos/file_1.jpg") {
		t.Errorf("redaction removed the file path, which is the useful part:\n%s", logs)
	}
}
