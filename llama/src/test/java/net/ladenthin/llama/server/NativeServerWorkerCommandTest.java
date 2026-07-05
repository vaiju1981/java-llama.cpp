// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.junit.jupiter.api.Assertions.assertThrows;

import net.ladenthin.llama.ClaudeGenerated;
import org.junit.jupiter.api.Test;

/**
 * Model-free unit tests for {@link NativeServer#setWorkerCommand(String...)} argument validation.
 * Only the rejection paths are tested here — they fire <em>before</em> the native library is
 * loaded, so this class runs on a pure-Java checkout. The accepted path (env round trip into the
 * router's worker spawn) is exercised by {@code RouterModeIntegrationTest}.
 */
@ClaudeGenerated(
        purpose = "Pin setWorkerCommand's fail-fast validation: tokens with whitespace, empty "
                + "tokens, null tokens and a null array must be rejected before any native call, "
                + "because the env variable is whitespace-split natively.")
public class NativeServerWorkerCommandTest {

    @Test
    public void tokenWithSpaceRejected() {
        assertThrows(IllegalArgumentException.class, () -> NativeServer.setWorkerCommand("java", "-cp", "a b.jar"));
    }

    @Test
    public void tokenWithTabRejected() {
        assertThrows(IllegalArgumentException.class, () -> NativeServer.setWorkerCommand("java\t-cp"));
    }

    @Test
    public void emptyTokenRejected() {
        assertThrows(IllegalArgumentException.class, () -> NativeServer.setWorkerCommand("java", ""));
    }

    @Test
    public void nullTokenRejected() {
        assertThrows(IllegalArgumentException.class, () -> NativeServer.setWorkerCommand("java", (String) null));
    }

    @Test
    public void nullArrayRejected() {
        assertThrows(NullPointerException.class, () -> NativeServer.setWorkerCommand((String[]) null));
    }
}
