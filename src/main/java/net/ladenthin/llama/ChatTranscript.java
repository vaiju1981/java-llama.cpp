// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import lombok.ToString;
import org.jspecify.annotations.Nullable;

/**
 * Append-only transcript of a multi-turn chat conversation, with an optional
 * leading {@code system} message. Extracted from {@link Session} so the
 * transcript invariants — especially the <b>two-phase commit</b> shape — are
 * testable independently of {@link LlamaModel} and its native library.
 *
 * <h2>Two-phase commit invariant</h2>
 *
 * <p>The append API only offers <b>atomic</b> turn commits:
 *
 * <ul>
 *   <li>{@link #appendRound(String, String)} appends a user turn AND an
 *       assistant turn in one synchronised operation — used by
 *       {@link Session#send(String)} on the model-success path. There is no
 *       way to commit only one half: if the model call throws, this method
 *       is simply never called and the transcript is untouched.</li>
 *   <li>{@link #appendUserTurn(String)} appends only the user turn — used
 *       by {@link Session#stream(String)} when the streaming iterable has
 *       been successfully created but the assistant reply is still being
 *       accumulated. The matching assistant turn is appended later via
 *       {@link #appendAssistantTurn(String)}.</li>
 * </ul>
 *
 * <p>The wire-format the model sees is built by
 * {@link #messagesWithPendingUserTurn(String)}, which returns a fresh list
 * containing the committed turns plus a pending user turn — <b>without
 * mutating</b> the underlying transcript. This is the mechanism by which the
 * model receives the prompt before the user turn is committed.
 *
 * <h2>Thread safety</h2>
 *
 * <p>This class is <b>not</b> internally synchronised. {@link Session} owns
 * the single instance and serialises access via its intrinsic lock, so the
 * transcript itself does not need additional synchronisation. Callers that
 * use {@code ChatTranscript} directly must provide their own synchronisation
 * if shared across threads.
 *
 * <h2>{@code toString} contract</h2>
 *
 * <p>Lombok-generated over the system message and turns list. The turns list
 * IS included because it is the operationally interesting state for log
 * traces. {@code equals}/{@code hashCode} are intentionally NOT generated:
 * a transcript instance is identified by its lifecycle owner ({@link Session}),
 * not by its accumulated content.
 */
@ToString
final class ChatTranscript {

    private final @Nullable String systemMessage;
    private final List<Pair<String, String>> turns = new ArrayList<Pair<String, String>>();

    /**
     * Create a new empty transcript with an optional system message.
     *
     * @param systemMessage the system prompt to prepend to every wire-format
     *     prompt; {@code null} or empty means "no system message"
     */
    ChatTranscript(@Nullable String systemMessage) {
        this.systemMessage = systemMessage;
    }

    /**
     * Append a user turn AND an assistant turn atomically. This is the only
     * API that records both halves of a round, so the two-phase commit
     * invariant is enforced by construction: callers that observe a model
     * call failure simply never invoke this method.
     *
     * @param userMessage the user turn
     * @param assistantMessage the assistant reply that completes the round
     */
    void appendRound(String userMessage, String assistantMessage) {
        turns.add(new Pair<String, String>("user", userMessage));
        turns.add(new Pair<String, String>("assistant", assistantMessage));
    }

    /**
     * Append a user turn. Used by streaming flows where the assistant reply
     * is accumulated incrementally and committed later via
     * {@link #appendAssistantTurn(String)}.
     *
     * @param userMessage the user turn
     */
    void appendUserTurn(String userMessage) {
        turns.add(new Pair<String, String>("user", userMessage));
    }

    /**
     * Append an assistant turn. Used to complete a round that was begun
     * with {@link #appendUserTurn(String)}.
     *
     * @param assistantMessage the assistant reply
     */
    void appendAssistantTurn(String assistantMessage) {
        turns.add(new Pair<String, String>("assistant", assistantMessage));
    }

    /**
     * Build the wire-format messages list with a pending user turn appended,
     * <b>without mutating</b> this transcript. This is the snapshot a model
     * call receives before the user turn is committed; if the model call
     * fails, the pending turn evaporates and the transcript stays untouched.
     *
     * @param pendingUserMessage the user turn to include in the wire format
     * @return a fresh list containing the committed turns followed by the
     *     pending user turn
     */
    List<Pair<String, String>> messagesWithPendingUserTurn(String pendingUserMessage) {
        List<Pair<String, String>> wire = new ArrayList<Pair<String, String>>(turns.size() + 1);
        wire.addAll(turns);
        wire.add(new Pair<String, String>("user", pendingUserMessage));
        return wire;
    }

    /**
     * Return the system message, or {@code null} when none was configured.
     *
     * @return the system prompt, or {@code null}
     */
    @Nullable
    String getSystemMessage() {
        return systemMessage;
    }

    /**
     * Return an unmodifiable {@link ChatMessage} snapshot of the transcript,
     * including the system message if one was configured.
     *
     * @return the unmodifiable snapshot
     */
    List<ChatMessage> snapshot() {
        List<ChatMessage> out = new ArrayList<ChatMessage>(turns.size() + 1);
        if (systemMessage != null && !systemMessage.isEmpty()) {
            out.add(new ChatMessage("system", systemMessage));
        }
        for (Pair<String, String> p : turns) {
            out.add(new ChatMessage(p.getKey(), p.getValue()));
        }
        return Collections.unmodifiableList(out);
    }

    /**
     * Return the number of committed turns (user + assistant). Does NOT
     * include the system message.
     *
     * @return the turn count
     */
    int size() {
        return turns.size();
    }
}
