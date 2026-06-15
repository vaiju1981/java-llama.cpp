// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
package net.ladenthin.llama;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import java.util.stream.Collectors;
import net.ladenthin.llama.value.ChatMessage;
import org.junit.jupiter.api.Test;

/**
 * Model-free unit tests pinning the {@link SessionState} contract extracted from
 * {@link Session}: the two-phase transcript commit and the streaming guard. These
 * run in the ordinary suite (no native library), complementing the model-gated
 * {@code SessionConcurrencyTest} and the agent-driven
 * {@code SessionStateInterleavingTest}.
 */
public class SessionStateTest {

    private static List<String> roles(List<ChatMessage> messages) {
        return messages.stream().map(ChatMessage::getRole).collect(Collectors.toList());
    }

    @Test
    public void send_commitsRound_andReturnsReply() {
        SessionState state = new SessionState(0, null);

        String reply = state.send("hello", (systemMessage, wireMessages) -> "hi there");

        assertThat(reply, is("hi there"));
        assertThat(roles(state.snapshot()), contains("user", "assistant"));
    }

    @Test
    public void send_passesSystemMessageAndPendingUserTurn_toModelCall() {
        SessionState state = new SessionState(0, "be terse");

        state.send("first", (systemMessage, wireMessages) -> {
            assertThat(systemMessage, is("be terse"));
            // wire format carries the pending user turn before it is committed
            assertThat(wireMessages.get(wireMessages.size() - 1).getValue(), is("first"));
            return "ok";
        });

        assertThat(roles(state.snapshot()), contains("system", "user", "assistant"));
    }

    @Test
    public void beginStream_thenCommit_completesRound_andClearsGuard() {
        SessionState state = new SessionState(0, null);

        String handle = state.beginStream("q", (systemMessage, wireMessages) -> "HANDLE");
        assertThat(handle, is("HANDLE"));
        // only the user turn is committed until the reply is committed
        assertThat(roles(state.snapshot()), contains("user"));

        state.commitStreamedReply("a");
        assertThat(roles(state.snapshot()), contains("user", "assistant"));

        // guard cleared: a follow-up send succeeds
        state.send("again", (systemMessage, wireMessages) -> "b");
        assertThat(roles(state.snapshot()), contains("user", "assistant", "user", "assistant"));
    }

    @Test
    public void send_whileStreaming_throwsWithDiagnosticMessage() {
        SessionState state = new SessionState(7, null);
        state.beginStream("q", (systemMessage, wireMessages) -> "HANDLE");

        IllegalStateException ex = assertThrows(IllegalStateException.class, () -> state.send("x", (s, w) -> "y"));
        assertThat(ex.getMessage(), containsString("stream in progress on slot 7"));
        assertThat(ex.getMessage(), containsString("before send(...)"));
    }

    @Test
    public void runWhenNotStreaming_throwsWhileStreaming_andRunsOtherwise() {
        SessionState state = new SessionState(3, null);

        assertThat(state.runWhenNotStreaming("save", () -> "saved"), is("saved"));

        state.beginStream("q", (systemMessage, wireMessages) -> "HANDLE");
        IllegalStateException ex =
                assertThrows(IllegalStateException.class, () -> state.runWhenNotStreaming("restore", () -> "restored"));
        assertThat(ex.getMessage(), containsString("before restore(...)"));
    }

    @Test
    public void commitStreamedReply_withoutStream_throws() {
        SessionState state = new SessionState(0, null);

        IllegalStateException ex = assertThrows(IllegalStateException.class, () -> state.commitStreamedReply("a"));
        assertThat(ex.getMessage(), containsString("no stream in progress"));
    }

    @Test
    public void runUnderLock_runsActionRegardlessOfStreamingState() {
        SessionState state = new SessionState(0, null);
        state.beginStream("q", (systemMessage, wireMessages) -> "HANDLE");

        boolean[] ran = {false};
        state.runUnderLock(() -> ran[0] = true);

        assertThat(ran[0], is(true));
    }
}
