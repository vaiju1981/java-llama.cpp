// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.exception;

import static org.junit.jupiter.api.Assertions.*;

import net.ladenthin.llama.ClaudeGenerated;
import org.junit.jupiter.api.Test;

@ClaudeGenerated(
        purpose = "Verify the typed-exception unification shape of ModelUnavailableException: the "
                + "(message) and (message, cause) constructor matrix, that it is a typed subclass of "
                + "LlamaException (so callers can catch it by the common base), and that it can be "
                + "thrown and caught.")
public class ModelUnavailableExceptionTest {

    @Test
    public void testMessageIsPreserved() {
        ModelUnavailableException ex = new ModelUnavailableException("model file missing");
        assertEquals("model file missing", ex.getMessage());
    }

    @Test
    public void testMessageAndCausePreserved() {
        Throwable cause = new IllegalStateException("skip-download set");
        ModelUnavailableException ex = new ModelUnavailableException("model file missing", cause);
        assertEquals("model file missing", ex.getMessage());
        assertSame(cause, ex.getCause());
    }

    @Test
    public void testIsLlamaException() {
        ModelUnavailableException ex = new ModelUnavailableException("error");
        assertTrue(ex instanceof LlamaException);
    }

    @Test
    public void testIsRuntimeException() {
        ModelUnavailableException ex = new ModelUnavailableException("error");
        assertTrue(ex instanceof RuntimeException);
    }

    @Test
    public void testNullMessage() {
        ModelUnavailableException ex = new ModelUnavailableException(null);
        assertNull(ex.getMessage());
    }

    @Test
    public void testCanBeCaughtAsLlamaException() {
        boolean caught = false;
        try {
            throw new ModelUnavailableException("thrown");
        } catch (LlamaException e) {
            assertEquals("thrown", e.getMessage());
            caught = true;
        }
        assertTrue(caught, "Expected ModelUnavailableException to be catchable as LlamaException");
    }
}
