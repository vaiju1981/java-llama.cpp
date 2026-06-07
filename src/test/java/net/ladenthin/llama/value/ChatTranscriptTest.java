// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.value;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.empty;
import static org.hamcrest.Matchers.hasItem;
import static org.hamcrest.Matchers.hasSize;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.not;
import static org.hamcrest.Matchers.nullValue;
import static org.hamcrest.Matchers.sameInstance;
import static org.junit.jupiter.api.Assertions.assertThrows;

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
        assertThat("wire-format must carry the pending user turn", wire, hasItem(new Pair<>("user", userMessage)));
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

            assertThat(t.size(), is(2));
            List<ChatMessage> snapshot = t.snapshot();
            assertThat(snapshot, hasSize(2));
            assertThat(snapshot.get(0).getRole(), is("user"));
            assertThat(snapshot.get(0).getContent(), is("hi"));
            assertThat(snapshot.get(1).getRole(), is("assistant"));
            assertThat(snapshot.get(1).getContent(), is("hello back"));
        }

        @Test
        @DisplayName("appendUserTurn + appendAssistantTurn together produce the same shape as appendRound")
        void appendUserAndAssistantSeparatelyMatchAppendRound() {
            ChatTranscript a = new ChatTranscript(null);
            ChatTranscript b = new ChatTranscript(null);

            a.appendRound("hi", "hello back");
            b.appendUserTurn("hi");
            b.appendAssistantTurn("hello back");

            assertThat("atomic-round and split-commit must converge", b.snapshot(), is(a.snapshot()));
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
            assertThat("1 user + 1 assistant + 1 pending user", wire, hasSize(3));
            assertThat(wire.get(2).getKey(), is("user"));
            assertThat(wire.get(2).getValue(), is("pending"));

            // The transcript itself MUST be unchanged.
            assertThat("transcript size unchanged", t.size(), is(sizeBefore));
            assertThat("transcript snapshot unchanged", t.snapshot(), is(snapshotBefore));
        }

        @Test
        @DisplayName("messagesWithPendingUserTurn returns a fresh list each call")
        void messagesWithPendingUserTurnReturnsFreshList() {
            ChatTranscript t = new ChatTranscript(null);
            List<Pair<String, String>> first = t.messagesWithPendingUserTurn("hi");
            List<Pair<String, String>> second = t.messagesWithPendingUserTurn("hi");
            assertThat(
                    "each wire-format build returns a fresh list — callers may mutate without affecting peers",
                    first,
                    is(not(sameInstance(second))));
        }

        @Test
        @DisplayName("snapshot includes system message when configured")
        void snapshotIncludesSystemMessage() {
            ChatTranscript t = new ChatTranscript("you are an assistant");
            t.appendRound("hi", "hello");

            List<ChatMessage> snap = t.snapshot();

            assertThat(snap, hasSize(3));
            assertThat(snap.get(0).getRole(), is("system"));
            assertThat(snap.get(0).getContent(), is("you are an assistant"));
        }

        @Test
        @DisplayName("snapshot omits system message when null or empty")
        void snapshotOmitsSystemMessageWhenAbsent() {
            assertThat(new ChatTranscript(null).snapshot(), is(empty()));
            assertThat(new ChatTranscript("").snapshot(), is(empty()));
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
            assertThat(new ChatTranscript(null).getSystemMessage(), is(nullValue()));
        }
    }

    @Nested
    @DisplayName("two-phase commit pattern — running documentation")
    class TwoPhaseCommit {

        @Test
        @DisplayName("simulated model failure leaves a FRESH transcript untouched")
        void freshTranscriptUntouchedWhenModelThrows() {
            ChatTranscript t = new ChatTranscript("system");
            assertThat("precondition: fresh transcript has no turns", t.size(), is(0));
            int snapshotSizeBefore = t.snapshot().size();

            // Caller simulates Session.send where the model rejects the request.
            assertThrows(
                    LlamaException.class,
                    () -> simulateSendThatModelRejects(
                            t, "first attempt", new LlamaException("simulated model failure")));

            // Two-phase commit: the pending user turn never landed in the transcript.
            // (The system message snapshot entry was there before and is still there.)
            assertThat("transcript MUST NOT contain the pending user turn after model failure", t.size(), is(0));
            assertThat(
                    "snapshot size unchanged by the failed call", t.snapshot().size(), is(snapshotSizeBefore));
        }

        @Test
        @DisplayName("simulated model failure leaves an EXISTING transcript byte-for-byte unchanged")
        void existingTranscriptUntouchedWhenModelThrows() {
            ChatTranscript t = new ChatTranscript("system");
            simulateSend(t, "hi", "hello back");
            simulateSend(t, "how are you", "i'm fine");

            List<ChatMessage> before = t.snapshot();
            assertThat("precondition: 1 system + 2 user + 2 assistant", before, hasSize(5));

            // Now the model rejects a third call.
            assertThrows(
                    LlamaException.class,
                    () -> simulateSendThatModelRejects(
                            t, "third attempt", new LlamaException("simulated model failure")));

            // Two-phase commit: existing transcript is byte-for-byte unchanged.
            List<ChatMessage> after = t.snapshot();
            assertThat("failed call must leave the transcript byte-for-byte unchanged", after, is(before));
        }

        @Test
        @DisplayName("simulated model success commits user + assistant atomically — never just one half")
        void successCommitsBothTurnsAtomically() {
            ChatTranscript t = new ChatTranscript(null);

            simulateSend(t, "hi", "hello");

            assertThat("both turns committed", t.size(), is(2));
            // The shape is invariant: there is no API to commit only one half via appendRound.
            // Spot-check that the turn pair is well-formed.
            List<ChatMessage> snap = t.snapshot();
            assertThat(snap.get(0).getRole(), is("user"));
            assertThat(snap.get(0).getContent(), is("hi"));
            assertThat(snap.get(1).getRole(), is("assistant"));
            assertThat(snap.get(1).getContent(), is("hello"));
        }

        @Test
        @DisplayName("stream() shape — user turn only, assistant follows via commitStreamedReply")
        void streamShape() {
            ChatTranscript t = new ChatTranscript(null);

            // Phase 1: build wire format (would be passed to model.generateChat).
            List<Pair<String, String>> wire = t.messagesWithPendingUserTurn("tell me a joke");
            assertThat("wire contains the pending user turn", wire, hasSize(1));

            // Phase 2: model returned an iterable successfully — commit only the user turn.
            t.appendUserTurn("tell me a joke");
            assertThat("user turn committed; assistant follows later", t.size(), is(1));

            // Later: caller invoked commitStreamedReply with the accumulated text.
            t.appendAssistantTurn("knock knock");
            assertThat("round closes with the assistant turn", t.size(), is(2));
            assertThat(t.snapshot().get(1).getRole(), is("assistant"));
        }
    }
}
