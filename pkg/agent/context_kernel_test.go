package agent

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeKernelFile(t *testing.T, workspace, content string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(workspace, kernelIdentityFile), []byte(content), 0o600); err != nil {
		t.Fatalf("write %s: %v", kernelIdentityFile, err)
	}
}

func TestGetIdentity_NoKernelFile_UsesBuiltinIdentity(t *testing.T) {
	cb := NewContextBuilder(t.TempDir())

	identity := cb.getIdentity(true)

	if !strings.Contains(identity, "You are picoclaw, a helpful AI assistant.") {
		t.Fatalf("expected builtin identity without KERNEL.md, got:\n%s", identity)
	}
}

func TestGetIdentity_KernelFile_ReplacesBuiltinIdentity(t *testing.T) {
	workspace := t.TempDir()
	const kernel = "You are MiteClaw, Bean's personal assistant.\n"
	writeKernelFile(t, workspace, kernel)

	identity := NewContextBuilder(workspace).getIdentity(true)

	if identity != kernel {
		t.Fatalf("expected identity to be KERNEL.md content, got:\n%s", identity)
	}
	if strings.Contains(identity, "You are picoclaw, a helpful AI assistant.") {
		t.Fatalf("builtin identity leaked into system prompt:\n%s", identity)
	}
}

func TestGetIdentity_BlankKernelFile_FallsBackToBuiltinIdentity(t *testing.T) {
	workspace := t.TempDir()
	writeKernelFile(t, workspace, "   \n\t\n")

	identity := NewContextBuilder(workspace).getIdentity(true)

	if !strings.Contains(identity, "You are picoclaw, a helpful AI assistant.") {
		t.Fatalf("expected builtin identity for blank KERNEL.md, got:\n%s", identity)
	}
}
