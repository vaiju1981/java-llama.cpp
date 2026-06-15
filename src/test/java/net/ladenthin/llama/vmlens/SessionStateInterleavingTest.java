// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
package net.ladenthin.llama.vmlens;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.nullValue;

import com.vmlens.api.AllInterleavings;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import net.ladenthin.llama.SessionState;
import net.ladenthin.llama.value.ChatMessage;
import org.junit.jupiter.api.Test;

/**
 * vmlens interleaving analysis of the {@link SessionState} streaming-guard and
 * two-phase transcript commit.
 *
 * <p>One thread runs a synchronous {@code send} round while another runs a
 * {@code beginStream} followed by {@code commitStreamedReply}. Both mutate the
 * shared {@code streamingActive} flag and the transcript under {@code SessionState}'s
 * single lock. The compound-atomicity invariant that must hold under <em>every</em>
 * interleaving: the resulting transcript is always a strictly alternating
 * {@code user, assistant, …} sequence (no torn round, no two same-role turns in a
 * row), the streaming flag is never left stuck (a follow-up {@code send} succeeds),
 * and the only acceptable failure is the {@link IllegalStateException} a {@code send}
 * raises when it loses the race to an in-progress stream.</p>
 *
 * <p>This is the multi-variable check-then-act ordering class that the existing
 * single-{@code volatile} {@code CancellationToken} Lincheck/jcstress tests do not
 * cover and that the model-gated {@code SessionConcurrencyTest} cannot explore
 * exhaustively. The model call is injected as a pure-Java lambda, so no native
 * library is needed. Like the rest of the package it runs only under the vmlens
 * agent (see the {@code vmlens} profile and the {@code maven-surefire-plugin}
 * {@code <excludes>} in {@code pom.xml}).</p>
 *
 * <p>Raw {@link Thread} usage is intentional: vmlens explores the interleavings of
 * the threads it directly manages.</p>
 */
public class SessionStateInterleavingTest {

    /**
     * Drives a synchronous send against a stream round through every interleaving and
     * asserts strict user/assistant alternation plus a non-stuck streaming flag.
     *
     * @throws InterruptedException if joining a worker thread is interrupted
     */
    @Test
    public void sendRacingStreamKeepsStrictAlternation() throws InterruptedException {
        try (AllInterleavings allInterleavings = new AllInterleavings("SessionState.streamGuard")) {
            while (allInterleavings.hasNext()) {
                final SessionState state = new SessionState(0, null);
                state.send("u0", (systemMessage, wireMessages) -> "a0"); // seed one aligned round

                final AtomicReference<Throwable> failure = new AtomicReference<>();

                final Thread sender = new Thread(() -> {
                    try {
                        state.send("u1", (systemMessage, wireMessages) -> "a1");
                    } catch (IllegalStateException lostRaceToStream) {
                        // Acceptable: a stream was in progress when send checked the guard.
                    } catch (Throwable t) {
                        failure.compareAndSet(null, t);
                    }
                });
                final Thread streamer = new Thread(() -> {
                    try {
                        state.beginStream("u2", (systemMessage, wireMessages) -> "streamHandle");
                        state.commitStreamedReply("a2");
                    } catch (Throwable t) {
                        failure.compareAndSet(null, t);
                    }
                });

                sender.start();
                streamer.start();
                sender.join();
                streamer.join();

                assertThat(failure.get(), is(nullValue()));
                assertStrictAlternation(state.snapshot());
                // The streaming flag must be cleared: a follow-up send must not throw.
                state.send("u3", (systemMessage, wireMessages) -> "a3");
                assertStrictAlternation(state.snapshot());
            }
        }
    }

    private static void assertStrictAlternation(List<ChatMessage> messages) {
        for (int i = 0; i < messages.size(); i++) {
            final String expectedRole = (i % 2 == 0) ? "user" : "assistant";
            assertThat("role at index " + i, messages.get(i).getRole(), is(expectedRole));
        }
    }
}
