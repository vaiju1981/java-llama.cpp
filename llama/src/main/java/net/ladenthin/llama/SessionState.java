// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Supplier;
import lombok.ToString;
import net.ladenthin.llama.value.ChatMessage;
import net.ladenthin.llama.value.ChatTranscript;
import net.ladenthin.llama.value.Pair;
import org.jspecify.annotations.Nullable;

/**
 * The lock-guarded conversation state machine behind {@link Session}: a
 * {@code streamingActive} flag plus a {@link ChatTranscript}, all serialised on a
 * single intrinsic lock. Extracted from {@link Session} so the concurrency contract
 * — the compound atomicity of the streaming guard and the two-phase transcript
 * commit — is testable independently of {@link LlamaModel} and its native library
 * (the same testability rationale that produced {@link ChatTranscript}).
 *
 * <p>The native model call is injected as a callback that runs <em>under the
 * lock</em>, between the guard check and the transcript commit, so {@link Session}
 * keeps exactly its previous serialisation semantics while this class owns no model
 * state and can be exercised with plain lambdas.</p>
 *
 * <h2>Compound-atomicity invariant</h2>
 *
 * <p>Under every interleaving of concurrent callers, the {@code streamingActive}
 * flag and the transcript move together: {@link #send(String, BiFunction)} and
 * {@link #beginStream(String, BiFunction)} fail-fast with
 * {@link IllegalStateException} while a stream is in progress, and the transcript
 * therefore always stays a strictly alternating {@code user, assistant, …}
 * sequence. This is the property the {@code SessionStateInterleavingTest} pins under
 * vmlens.</p>
 *
 * <h2>{@code toString} contract</h2>
 *
 * <p>Lombok-generated over the slot id, transcript, and streaming flag. The
 * intrinsic {@code lock} is excluded as a noise field. {@code equals}/{@code
 * hashCode} are intentionally NOT generated: like {@link Session} this is a mutable
 * lifecycle handle identified by identity.</p>
 */
@ToString
public final class SessionState {

    // Intrinsic lock used only for synchronisation; rendering its identity adds noise.
    @ToString.Exclude
    private final Object lock = new Object();

    private final int slotId;

    private final ChatTranscript transcript;

    private boolean streamingActive;

    /**
     * Create a new state machine for the given slot, with an optional system prompt.
     *
     * @param slotId the slot id, used only to render diagnostic messages
     * @param systemMessage optional system prompt (may be {@code null} or empty)
     */
    public SessionState(int slotId, @Nullable String systemMessage) {
        this.slotId = slotId;
        this.transcript = new ChatTranscript(systemMessage);
    }

    /**
     * Run a synchronous send round atomically: fail-fast if a stream is in
     * progress, invoke {@code modelCall} (under the lock) with the system message
     * and the wire-format messages carrying a pending user turn, then commit the
     * user turn and the returned reply as one round.
     *
     * @param userMessage the user turn to commit on success
     * @param modelCall the model invocation; receives {@code (systemMessage,
     *     wireMessages)} and returns the assistant reply text
     * @return the assistant reply produced by {@code modelCall}
     * @throws IllegalStateException if a stream is currently in progress
     */
    public String send(String userMessage, BiFunction<@Nullable String, List<Pair<String, String>>, String> modelCall) {
        synchronized (lock) {
            requireNotStreaming("send");
            String reply =
                    modelCall.apply(transcript.getSystemMessage(), transcript.messagesWithPendingUserTurn(userMessage));
            transcript.appendRound(userMessage, reply);
            return reply;
        }
    }

    /**
     * Begin a streaming round atomically: fail-fast if a stream is in progress,
     * invoke {@code modelCall} (under the lock) to obtain the stream handle, then
     * commit the user turn and mark streaming active. The matching assistant turn
     * is committed later via {@link #commitStreamedReply(String)}.
     *
     * @param <R> the stream-handle type returned by {@code modelCall}
     * @param userMessage the user turn to commit on success
     * @param modelCall the model invocation; receives {@code (systemMessage,
     *     wireMessages)} and returns the stream handle
     * @return the stream handle produced by {@code modelCall}
     * @throws IllegalStateException if a stream is currently in progress
     */
    public <R> R beginStream(
            String userMessage, BiFunction<@Nullable String, List<Pair<String, String>>, R> modelCall) {
        synchronized (lock) {
            requireNotStreaming("stream");
            R streamHandle =
                    modelCall.apply(transcript.getSystemMessage(), transcript.messagesWithPendingUserTurn(userMessage));
            transcript.appendUserTurn(userMessage);
            streamingActive = true;
            return streamHandle;
        }
    }

    /**
     * Commit the assistant reply accumulated from a prior
     * {@link #beginStream(String, BiFunction)} and clear the streaming flag.
     *
     * @param assistantText the assistant text to append
     * @throws IllegalStateException if no stream is currently in progress
     */
    public void commitStreamedReply(String assistantText) {
        synchronized (lock) {
            if (!streamingActive) {
                throw new IllegalStateException("no stream in progress on slot " + slotId
                        + " (transcript=" + transcript.size() + " turns)"
                        + "; call stream(...) first");
            }
            transcript.appendAssistantTurn(assistantText);
            streamingActive = false;
        }
    }

    /**
     * Abandon an in-progress streaming round without recording an assistant reply: clears the
     * streaming flag and rolls back the pending user turn, returning the state to its pre-stream
     * shape. Safe to call when no stream is active (no-op), so it can run in a {@code finally}
     * block to guarantee the session never stays wedged after an abandoned or failed stream.
     */
    public void cancelStream() {
        synchronized (lock) {
            if (streamingActive) {
                transcript.removePendingUserTurn();
                streamingActive = false;
            }
        }
    }

    /**
     * Run an action under the lock, but only when no stream is in progress
     * (used for slot save/restore, which must not race a streaming round).
     *
     * @param <R> the action result type
     * @param operation the operation name used in the diagnostic message
     * @param action the action to run while holding the lock
     * @return the action result
     * @throws IllegalStateException if a stream is currently in progress
     */
    public <R> R runWhenNotStreaming(String operation, Supplier<R> action) {
        synchronized (lock) {
            requireNotStreaming(operation);
            return action.get();
        }
    }

    /**
     * Run an action under the lock with no streaming guard (used for slot erase on
     * {@code close()}, which is valid regardless of streaming state).
     *
     * @param action the action to run while holding the lock
     */
    public void runUnderLock(Runnable action) {
        synchronized (lock) {
            action.run();
        }
    }

    /**
     * Return an unmodifiable snapshot of the transcript, including the system
     * message if one was configured.
     *
     * @return the unmodifiable transcript snapshot
     */
    public List<ChatMessage> snapshot() {
        synchronized (lock) {
            return transcript.snapshot();
        }
    }

    /**
     * Return a fresh copy of the committed (role, text) turns for checkpointing. Rejected
     * while a stream is in progress: the pending user turn is already committed at that
     * point, so a mid-stream snapshot would capture a dangling half-round.
     *
     * @return a fresh copy of the committed turns, in order
     * @throws IllegalStateException if a stream is in progress
     */
    public List<Pair<String, String>> turnsSnapshot() {
        synchronized (lock) {
            requireNotStreaming("checkpoint");
            return transcript.turnsSnapshot();
        }
    }

    /**
     * Replace the transcript's committed turns with a checkpointed snapshot, for
     * rewinding. Rejected while a stream is in progress.
     *
     * @param turns the (role, text) turns to restore, in order
     * @throws IllegalStateException if a stream is in progress
     */
    public void restoreTurns(List<Pair<String, String>> turns) {
        synchronized (lock) {
            requireNotStreaming("rewind");
            transcript.resetTurns(turns);
        }
    }

    /**
     * System-message accessor (fixed at construction).
     *
     * @return the system prompt, or {@code null} when none was configured
     */
    public @Nullable String getSystemMessage() {
        return transcript.getSystemMessage();
    }

    private void requireNotStreaming(String operation) {
        if (streamingActive) {
            throw new IllegalStateException("stream in progress on slot " + slotId
                    + " (transcript=" + transcript.size() + " turns)"
                    + "; call commitStreamedReply(...) before " + operation + "(...)");
        }
    }
}
