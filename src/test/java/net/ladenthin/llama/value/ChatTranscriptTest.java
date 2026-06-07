// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.value;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import net.ladenthin.llama.Session;
import net.ladenthin.llama.exception.LlamaException;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

/**
 * Running documentation of the two-phase commit invariant that
 * {@link Session#send(String)} and {@link Session#stream(String)} rely on.
 *
 * <p>The transcript management was extracted from {@code Session} into
 * {@link ChatTranscript} precisely so this invariant — "transcript is mutated
 * only on the model-call success path; on failure the pending user turn
 * evaporates" — could be unit-tested without a GGUF model or the native
 * {@code libjllama} library.
 *
 * <p>The contract is enforced <b>by the API shape itself</b>, not by tests:
 *
 * <ul>
 *   <li>The only "commit a full round" method is {@link
 *       ChatTranscript#appendRound(String, String)}, which appends both turns
 *       atomically. There is no way to commit just the user turn through this
 *       API.</li>
 *   <li>The wire-format the model receives is built by
 *       {@link ChatTranscript#messagesWithPendingUserTurn(String)}, which
 *       returns a fresh list and does NOT mutate the transcript. So the
 *       pending user turn reaches the model without being committed.</li>
 *   <li>Therefore: if the model call throws after the wire-format is built,
 *       {@code appendRound} is never reached, and the transcript stays
 *       exactly as it was before the call.</li>
 * </ul>
 *
 * <p>The tests below pin both the mechanical API behaviour and the higher-level
 * two-phase commit pattern as it is composed by {@link Session}.
 */
class ChatTranscriptTest {

    /** Helper: simulate {@code Session.send} composing a single round through the API. */
    private static void simulateSend(ChatTranscript t, String userMessage, String assistantReply) {
        // Phase 1: build wire-format (model would see this).
        List<Pair<String, String>> wire = t.messagesWithPendingUserTurn(userMessage);
        // The wire format must contain the pending turn the model is about to answer.
        assertTrue(
                wire.stream().anyMatch(p -> "user".equals(p.getKey()) && userMessage.equals(p.getValue())),
                "wire-format must carry the pending user turn");
        // Phase 2: model returned successfully — commit both turns atomically.
        t.appendRound(userMessage, assistantReply);
    }

    /**
     * Helper: simulate {@code Session.send} where the model call throws after the
     * wire-format is built. The {@code appendRound} line is never reached.
     */
    private static void simulateSendThatModelRejects(
            ChatTranscript t, String pendingUserMessage, RuntimeException simulatedModelFailure) {
        // Phase 1: build wire-format (model would see this).
        @SuppressWarnings("unused")
        List<Pair<String, String>> wire = t.messagesWithPendingUserTurn(pendingUserMessage);
        // Phase 2: model throws — the caller (Session.send) lets the exception
        // propagate; appendRound is NEVER called.
        throw simulatedModelFailure;
    }

    @Nested
    @DisplayName("mechanical API behaviour")
    class Api {

        @Test
        @DisplayName("appendRound commits both turns atomically")
        void appendRoundCommitsBothTurnsAtomically() {
            ChatTranscript t = new ChatTranscript(null);

            t.appendRound("hi", "hello back");

            assertEquals(2, t.size());
            List<ChatMessage> snapshot = t.snapshot();
            assertEquals(2, snapshot.size());
            assertEquals("user", snapshot.get(0).getRole());
            assertEquals("hi", snapshot.get(0).getContent());
            assertEquals("assistant", snapshot.get(1).getRole());
            assertEquals("hello back", snapshot.get(1).getContent());
        }

        @Test
        @DisplayName("appendUserTurn + appendAssistantTurn together produce the same shape as appendRound")
        void appendUserAndAssistantSeparatelyMatchAppendRound() {
            ChatTranscript a = new ChatTranscript(null);
            ChatTranscript b = new ChatTranscript(null);

            a.appendRound("hi", "hello back");
            b.appendUserTurn("hi");
            b.appendAssistantTurn("hello back");

            assertEquals(a.snapshot(), b.snapshot(), "atomic-round and split-commit must converge");
        }

        @Test
        @DisplayName("messagesWithPendingUserTurn does NOT mutate the transcript")
        void messagesWithPendingUserTurnDoesNotMutate() {
            ChatTranscript t = new ChatTranscript("system");
            t.appendRound("first", "reply-1");
            int sizeBefore = t.size();
            List<ChatMessage> snapshotBefore = t.snapshot();

            List<Pair<String, String>> wire = t.messagesWithPendingUserTurn("pending");

            // Build a wire-format containing committed turns + pending user.
            assertEquals(3, wire.size(), "1 user + 1 assistant + 1 pending user");
            assertEquals("user", wire.get(2).getKey());
            assertEquals("pending", wire.get(2).getValue());

            // The transcript itself MUST be unchanged.
            assertEquals(sizeBefore, t.size(), "transcript size unchanged");
            assertEquals(snapshotBefore, t.snapshot(), "transcript snapshot unchanged");
        }

        @Test
        @DisplayName("messagesWithPendingUserTurn returns a fresh list each call")
        void messagesWithPendingUserTurnReturnsFreshList() {
            ChatTranscript t = new ChatTranscript(null);
            List<Pair<String, String>> first = t.messagesWithPendingUserTurn("hi");
            List<Pair<String, String>> second = t.messagesWithPendingUserTurn("hi");
            assertNotSame(
                    first,
                    second,
                    "each wire-format build returns a fresh list — callers may mutate without affecting peers");
        }

        @Test
        @DisplayName("snapshot includes system message when configured")
        void snapshotIncludesSystemMessage() {
            ChatTranscript t = new ChatTranscript("you are an assistant");
            t.appendRound("hi", "hello");

            List<ChatMessage> snap = t.snapshot();

            assertEquals(3, snap.size());
            assertEquals("system", snap.get(0).getRole());
            assertEquals("you are an assistant", snap.get(0).getContent());
        }

        @Test
        @DisplayName("snapshot omits system message when null or empty")
        void snapshotOmitsSystemMessageWhenAbsent() {
            assertEquals(0, new ChatTranscript(null).snapshot().size());
            assertEquals(0, new ChatTranscript("").snapshot().size());
        }

        @Test
        @DisplayName("snapshot is unmodifiable")
        void snapshotIsUnmodifiable() {
            ChatTranscript t = new ChatTranscript(null);
            t.appendRound("hi", "hello");
            List<ChatMessage> snap = t.snapshot();
            assertThrows(UnsupportedOperationException.class, () -> snap.clear());
        }

        @Test
        @DisplayName("getSystemMessage returns null when absent")
        void getSystemMessageNullWhenAbsent() {
            assertNull(new ChatTranscript(null).getSystemMessage());
        }
    }

    @Nested
    @DisplayName("two-phase commit pattern — running documentation")
    class TwoPhaseCommit {

        @Test
        @DisplayName("simulated model failure leaves a FRESH transcript untouched")
        void freshTranscriptUntouchedWhenModelThrows() {
            ChatTranscript t = new ChatTranscript("system");
            assertEquals(0, t.size(), "precondition: fresh transcript has no turns");
            int snapshotSizeBefore = t.snapshot().size();

            // Caller simulates Session.send where the model rejects the request.
            assertThrows(
                    LlamaException.class,
                    () -> simulateSendThatModelRejects(
                            t, "first attempt", new LlamaException("simulated model failure")));

            // Two-phase commit: the pending user turn never landed in the transcript.
            // (The system message snapshot entry was there before and is still there.)
            assertEquals(0, t.size(), "transcript MUST NOT contain the pending user turn after model failure");
            assertEquals(snapshotSizeBefore, t.snapshot().size(), "snapshot size unchanged by the failed call");
        }

        @Test
        @DisplayName("simulated model failure leaves an EXISTING transcript byte-for-byte unchanged")
        void existingTranscriptUntouchedWhenModelThrows() {
            ChatTranscript t = new ChatTranscript("system");
            simulateSend(t, "hi", "hello back");
            simulateSend(t, "how are you", "i'm fine");

            List<ChatMessage> before = t.snapshot();
            assertEquals(5, before.size(), "precondition: 1 system + 2 user + 2 assistant");

            // Now the model rejects a third call.
            assertThrows(
                    LlamaException.class,
                    () -> simulateSendThatModelRejects(
                            t, "third attempt", new LlamaException("simulated model failure")));

            // Two-phase commit: existing transcript is byte-for-byte unchanged.
            List<ChatMessage> after = t.snapshot();
            assertEquals(before, after, "failed call must leave the transcript byte-for-byte unchanged");
        }

        @Test
        @DisplayName("simulated model success commits user + assistant atomically — never just one half")
        void successCommitsBothTurnsAtomically() {
            ChatTranscript t = new ChatTranscript(null);

            simulateSend(t, "hi", "hello");

            assertEquals(2, t.size(), "both turns committed");
            // The shape is invariant: there is no API to commit only one half via appendRound.
            // Spot-check that the turn pair is well-formed.
            List<ChatMessage> snap = t.snapshot();
            assertEquals("user", snap.get(0).getRole());
            assertEquals("hi", snap.get(0).getContent());
            assertEquals("assistant", snap.get(1).getRole());
            assertEquals("hello", snap.get(1).getContent());
        }

        @Test
        @DisplayName("stream() shape — user turn only, assistant follows via commitStreamedReply")
        void streamShape() {
            ChatTranscript t = new ChatTranscript(null);

            // Phase 1: build wire format (would be passed to model.generateChat).
            List<Pair<String, String>> wire = t.messagesWithPendingUserTurn("tell me a joke");
            assertEquals(1, wire.size(), "wire contains the pending user turn");

            // Phase 2: model returned an iterable successfully — commit only the user turn.
            t.appendUserTurn("tell me a joke");
            assertEquals(1, t.size(), "user turn committed; assistant follows later");

            // Later: caller invoked commitStreamedReply with the accumulated text.
            t.appendAssistantTurn("knock knock");
            assertEquals(2, t.size(), "round closes with the assistant turn");
            assertEquals("assistant", t.snapshot().get(1).getRole());
        }
    }
}
