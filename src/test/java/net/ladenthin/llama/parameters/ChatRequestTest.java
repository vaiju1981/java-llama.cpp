// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.parameters;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import net.ladenthin.llama.value.ToolDefinition;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

/**
 * Running documentation of the {@link ChatRequest} immutability + wither-pattern
 * contract. Every modification method returns a NEW request; the original is
 * never mutated. Two requests with the same content compare equal regardless
 * of identity.
 */
class ChatRequestTest {

    @Nested
    @DisplayName("immutability — every modifier returns a fresh instance")
    class Immutability {

        @Test
        void appendMessageReturnsNewInstance() {
            ChatRequest original = ChatRequest.empty();
            ChatRequest derived = original.appendMessage("user", "hi");
            assertNotSame(original, derived);
            assertEquals(0, original.getMessages().size(), "original is untouched");
            assertEquals(1, derived.getMessages().size(), "derived has the message");
        }

        @Test
        void appendToolReturnsNewInstance() {
            ChatRequest original = ChatRequest.empty();
            ChatRequest derived = original.appendTool(new ToolDefinition("echo", "Echo", "{}"));
            assertNotSame(original, derived);
            assertEquals(0, original.getTools().size());
            assertEquals(1, derived.getTools().size());
        }

        @Test
        void withToolChoiceReturnsNewInstance() {
            ChatRequest original = ChatRequest.empty();
            ChatRequest derived = original.withToolChoice("auto");
            assertNotSame(original, derived);
            assertFalse(original.getToolChoice().isPresent(), "original toolChoice unset");
            assertEquals("auto", derived.getToolChoice().orElseThrow());
        }

        @Test
        void withMaxToolRoundsReturnsNewInstance() {
            ChatRequest original = ChatRequest.empty();
            ChatRequest derived = original.withMaxToolRounds(2);
            assertNotSame(original, derived);
            assertEquals(ChatRequest.DEFAULT_MAX_TOOL_ROUNDS, original.getMaxToolRounds());
            assertEquals(2, derived.getMaxToolRounds());
        }

        @Test
        void withInferenceCustomizerReturnsNewInstance() {
            ChatRequest original = ChatRequest.empty();
            ChatRequest derived = original.withInferenceCustomizer(p -> p.withSeed(42));
            assertNotSame(original, derived);
        }

        @Test
        @DisplayName("chained derivations leave every intermediate untouched")
        void chainedDerivationsLeaveIntermediatesUntouched() {
            ChatRequest a = ChatRequest.empty();
            ChatRequest b = a.appendMessage("user", "hi");
            ChatRequest c = b.appendMessage("assistant", "hello");
            ChatRequest d = c.withMaxToolRounds(3);

            assertEquals(0, a.getMessages().size());
            assertEquals(1, b.getMessages().size());
            assertEquals(2, c.getMessages().size());
            assertEquals(2, d.getMessages().size());
            assertEquals(ChatRequest.DEFAULT_MAX_TOOL_ROUNDS, c.getMaxToolRounds());
            assertEquals(3, d.getMaxToolRounds());
        }

        @Test
        @DisplayName("the messages accessor returns an unmodifiable view")
        void messagesAccessorIsUnmodifiable() {
            ChatRequest req = ChatRequest.empty().appendMessage("user", "hi");
            assertThrows(
                    UnsupportedOperationException.class, () -> req.getMessages().clear());
        }

        @Test
        @DisplayName("the tools accessor returns an unmodifiable view")
        void toolsAccessorIsUnmodifiable() {
            ChatRequest req = ChatRequest.empty().appendTool(new ToolDefinition("e", "d", "{}"));
            assertThrows(
                    UnsupportedOperationException.class, () -> req.getTools().clear());
        }
    }

    @Nested
    @DisplayName("equality — value semantics")
    class Equality {

        @Test
        void twoEmptyRequestsAreEqual() {
            assertEquals(ChatRequest.empty(), ChatRequest.empty());
        }

        @Test
        void sameContentSameEquality() {
            ChatRequest a = ChatRequest.empty().appendMessage("user", "hi").withMaxToolRounds(3);
            ChatRequest b = ChatRequest.empty().appendMessage("user", "hi").withMaxToolRounds(3);
            assertEquals(a, b);
            assertEquals(a.hashCode(), b.hashCode());
        }

        @Test
        void differentMessagesNotEqual() {
            ChatRequest a = ChatRequest.empty().appendMessage("user", "hi");
            ChatRequest b = ChatRequest.empty().appendMessage("user", "bye");
            assertNotEquals(a, b);
        }

        @Test
        void differentMaxToolRoundsNotEqual() {
            ChatRequest a = ChatRequest.empty().withMaxToolRounds(2);
            ChatRequest b = ChatRequest.empty().withMaxToolRounds(3);
            assertNotEquals(a, b);
        }

        @Test
        @DisplayName(
                "the customiser is excluded from equality — two requests with the same content but different lambdas are equal")
        void customizerExcludedFromEquality() {
            ChatRequest a = ChatRequest.empty().withInferenceCustomizer(p -> p.withSeed(1));
            ChatRequest b = ChatRequest.empty().withInferenceCustomizer(p -> p.withSeed(2));
            assertEquals(a, b, "different lambda identities must NOT make the requests unequal");
        }
    }

    @Nested
    @DisplayName("validation")
    class Validation {

        @Test
        void withMaxToolRoundsRejectsZero() {
            assertThrows(
                    IllegalArgumentException.class, () -> ChatRequest.empty().withMaxToolRounds(0));
        }

        @Test
        void withMaxToolRoundsRejectsNegative() {
            assertThrows(
                    IllegalArgumentException.class, () -> ChatRequest.empty().withMaxToolRounds(-1));
        }

        @Test
        void emptyMessageIsTheCanonicalStartingPoint() {
            assertSame(ChatRequest.empty(), ChatRequest.empty(), "empty() is a cached singleton");
        }
    }

    @Nested
    @DisplayName("JSON-build helpers stay read-only")
    class JsonHelpers {

        @Test
        void buildMessagesJsonDoesNotMutate() {
            ChatRequest req = ChatRequest.empty().appendMessage("user", "hi");
            String json = req.buildMessagesJson();
            assertTrue(json.contains("\"user\""), json);
            assertEquals(1, req.getMessages().size(), "build did not mutate the messages list");
        }

        @Test
        void buildToolsJsonEmptyWhenNoTools() {
            assertFalse(ChatRequest.empty().buildToolsJson().isPresent());
        }
    }
}
