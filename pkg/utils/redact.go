// PicoClaw - Ultra-lightweight personal AI agent

package utils

import "regexp"

// telegramBotToken matches a bot token where it appears as the path segment
// right after /bot or /file/bot, which is the only place Telegram puts it:
// <base>/bot<token>/<method> for API calls and <base>/file/bot<token>/<path>
// for downloads. The host is not part of the pattern because a deployment may
// point at its own Bot API server.
//
// The token shape is telego's: digits, a colon, then 35 word characters. The
// trailing group is a boundary rather than a lookahead, which RE2 does not
// have, and it is put back by the replacement.
//
// The pattern is deliberately narrow. A string like
// "https://example.com/files/bot123:abc/report.pdf" must survive untouched:
// redaction that eats ordinary URLs trades a credential leak for logs nobody
// can debug with.
var telegramBotToken = regexp.MustCompile(`(/(?:file/)?bot)\d+:[\w-]{35}([/?#'"\s]|$)`)

// RedactTelegramBotToken removes the Telegram bot token from a string that is
// about to be logged.
//
// It is applied to whole strings rather than to a known field, because the copy
// that leaks is the one nobody planned: net/http puts the request URL into the
// error it returns, so a transport failure carries the token in its message.
// Redacting the URL field alone would leave that copy behind.
//
// The token is the credential for the whole bot: anyone who reads it from a log
// can send and receive messages as the operator's bot.
func RedactTelegramBotToken(s string) string {
	if s == "" {
		return s
	}
	return telegramBotToken.ReplaceAllString(s, "${1}<redacted>${2}")
}

// redactedError renders an error for logging with the bot token removed.
func redactedError(err error) string {
	if err == nil {
		return ""
	}
	return RedactTelegramBotToken(err.Error())
}
