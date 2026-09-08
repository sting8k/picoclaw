package utils

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/sipeed/picoclaw/pkg/logger"
)

// canaryToken is an invented token with the exact shape telego accepts: digits,
// a colon, then 35 word characters. It is the value every test below looks for
// in the logs; a real token must never appear in a repro or a report.
const canaryToken = "123456789:AAFcanary_do_not_use_this_value_xyz"

func TestCanaryHasTheShapeOfARealToken(t *testing.T) {
	digits, rest, found := strings.Cut(canaryToken, ":")
	if !found || len(digits) < 9 || len(rest) != 35 {
		t.Fatalf("canary %q does not have the token shape the redactor looks for", canaryToken)
	}
}

func TestRedactSecretsRemovesBotTokens(t *testing.T) {
	cases := map[string]string{
		"download URL": "https://api.telegram.org/file/bot" + canaryToken + "/photos/file_1.jpg",
		"api URL":      "https://api.telegram.org/bot" + canaryToken + "/getFile?file_id=abc",
		"transport error": fmt.Sprintf(
			`Get "https://api.telegram.org/file/bot%s/photos/file_1.jpg": dial tcp: lookup failed`,
			canaryToken,
		),
	}

	for name, input := range cases {
		t.Run(name, func(t *testing.T) {
			got := RedactTelegramBotToken(input)
			if strings.Contains(got, canaryToken) {
				t.Errorf("token survived redaction: %s", got)
			}
			if !strings.Contains(got, "bot<redacted>") {
				t.Errorf("redacted form is missing: %s", got)
			}
			// The rest of the URL is what makes a log useful.
			if strings.Contains(input, "file_1.jpg") && !strings.Contains(got, "file_1.jpg") {
				t.Errorf("redaction removed more than the credential: %s", got)
			}
		})
	}
}

func TestRedactSecretsLeavesOtherURLsAlone(t *testing.T) {
	for _, input := range []string{
		"https://example.com/files/report.pdf",
		"https://slack.example.com/download?robot=1",
		// Token-shaped, but not where Telegram puts a token: a file called
		// bot<something> on somebody else's server.
		"https://example.com/files/bot123:abc/report.pdf",
		"https://example.com/files/bot" + canaryToken + ".pdf",
		// Prose that happens to contain the word.
		"robot123:abcdef could not be reached",
		// The right place, the wrong shape: too short to be a token.
		"https://api.telegram.org/bot123:abc/getFile",
		"",
	} {
		if got := RedactTelegramBotToken(input); got != input {
			t.Errorf("RedactTelegramBotToken(%q) = %q, want it unchanged", input, got)
		}
	}
}

// A deployment may run its own Bot API server, so redaction cannot depend on
// the host.
func TestRedactSecretsWorksOnACustomBotAPIHost(t *testing.T) {
	for _, input := range []string{
		"https://telegram.internal.example.com/file/bot" + canaryToken + "/photos/file_1.jpg",
		"http://127.0.0.1:8081/bot" + canaryToken + "/getFile?file_id=abc",
		"https://api.telegram.org/bot" + canaryToken,
	} {
		got := RedactTelegramBotToken(input)
		if strings.Contains(got, canaryToken) {
			t.Errorf("token survived on a custom host: %s", got)
		}
		if !strings.Contains(got, "bot<redacted>") {
			t.Errorf("redacted form is missing: %s", got)
		}
	}
}

func TestRedactedErrorHandlesNil(t *testing.T) {
	if got := redactedError(nil); got != "" {
		t.Errorf("redactedError(nil) = %q, want empty", got)
	}
	err := errors.New(`Get "https://api.telegram.org/file/bot` + canaryToken + `/x.jpg": timeout`)
	if got := redactedError(err); strings.Contains(got, canaryToken) {
		t.Errorf("token survived: %s", got)
	}
}

// captureLogs routes the real logger to a file for the duration of a test and
// returns everything it wrote. Testing the helper alone would not prove the
// call sites use it.
func captureLogs(t *testing.T, run func()) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "picoclaw.log")
	if err := logger.EnableFileLogging(path); err != nil {
		t.Fatalf("EnableFileLogging: %v", err)
	}
	previous := logger.GetLevel()
	logger.SetLevel(logger.DEBUG)
	defer func() {
		logger.SetLevel(previous)
		logger.DisableFileLogging()
	}()

	run()

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read log: %v", err)
	}
	return string(data)
}

// Every way a download can end must keep the token out of the log.
func TestDownloadFileNeverLogsTheToken(t *testing.T) {
	served := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("payload"))
	}))
	defer served.Close()

	missing := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer missing.Close()

	dead := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	deadURL := dead.URL
	dead.Close() // nothing is listening: the transport error carries the URL

	// Each URL carries the canary the way a Telegram download URL carries the
	// bot token.
	cases := map[string]string{
		"success":         served.URL + "/file/bot" + canaryToken + "/photos/file_1.jpg",
		"non-200":         missing.URL + "/file/bot" + canaryToken + "/photos/file_2.jpg",
		"transport error": deadURL + "/file/bot" + canaryToken + "/photos/file_3.jpg",
		"invalid URL":     "://file/bot" + canaryToken + "/broken",
	}

	for name, url := range cases {
		t.Run(name, func(t *testing.T) {
			logs := captureLogs(t, func() {
				DownloadFile(url, "canary.jpg", DownloadOptions{LoggerPrefix: "telegram"})
			})
			if strings.Contains(logs, canaryToken) {
				t.Errorf("the bot token reached the log:\n%s", logs)
			}
			if name != "success" && !strings.Contains(logs, "bot<redacted>") {
				t.Errorf("a failed download logged nothing identifiable:\n%s", logs)
			}
		})
	}
}

// A non-Telegram download must keep logging its URL in full: redaction that
// hides ordinary URLs would trade one debugging problem for another.
func TestDownloadFileStillLogsOrdinaryURLs(t *testing.T) {
	missing := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer missing.Close()

	url := missing.URL + "/files/report.pdf"
	logs := captureLogs(t, func() {
		DownloadFile(url, "report.pdf", DownloadOptions{LoggerPrefix: "slack"})
	})
	if !strings.Contains(logs, "/files/report.pdf") {
		t.Errorf("an ordinary download URL was not logged:\n%s", logs)
	}

	// And the log is still parseable JSON, i.e. redaction did not corrupt it.
	for _, line := range strings.Split(strings.TrimSpace(logs), "\n") {
		if line == "" {
			continue
		}
		var parsed map[string]any
		if err := json.Unmarshal([]byte(line), &parsed); err != nil {
			t.Fatalf("log line is not JSON: %v\n%s", err, line)
		}
	}
}
