// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.File;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import reactor.core.publisher.Flux;
import reactor.core.scheduler.Schedulers;
import reactor.test.StepVerifier;

/**
 * Proves the documented "reactive integration" pattern from the README works
 * end-to-end without adding {@code org.reactivestreams} as a runtime dependency.
 *
 * <p>{@link LlamaIterable} implements {@code Iterable<LlamaOutput> & AutoCloseable},
 * so Project Reactor, RxJava 3, Kotlin coroutines {@code Flow}, and Akka Streams
 * all wrap it in a single statement (see README "Reactive integration"). This
 * test exercises the Reactor path because it is the most demanding contract —
 * backpressure via {@code request(n)} and AutoCloseable cancel propagation —
 * and the same contract underpins the other libraries' iterable adapters.
 *
 * <p>{@link #mockIterable_requestBackpressureAndCancelClose()} runs without a
 * GGUF model: it uses a fake iterable that tracks {@code close()} so the
 * Reactor wiring is verified deterministically on every CI run.
 *
 * <p>{@link #realModel_cancelPropagatesToNativeCompletion()} additionally
 * proves end-to-end native cancel via llama.cpp's {@code cancelCompletion}, but
 * is gated on a model file being present (same gating pattern as
 * {@code LlamaModelTest}).
 */
class ReactorIntegrationTest {

    /**
     * Mock-only contract test — runs every build. Asserts:
     * <ol>
     *   <li>Reactor honours backpressure: {@code request(n)} delivers at most
     *       {@code n} items, never more (no producer overrun).</li>
     *   <li>Reactor closes the {@link AutoCloseable} iterable on cancel — which
     *       is the wire by which {@code LlamaIterable.close()} → native
     *       {@code cancelCompletion} on real generations.</li>
     * </ol>
     */
    @Test
    void mockIterable_requestBackpressureAndCancelClose() {
        AtomicBoolean closed = new AtomicBoolean(false);
        List<LlamaOutput> tokens =
                Arrays.asList(out("a"), out("b"), out("c"), out("d"), out("e"));

        // Flux.fromIterable(iterable) does NOT auto-close AutoCloseable iterables on cancel —
        // the canonical Reactor pattern for that is Flux.using(supplier, builder, cleanup).
        // The cleanup runs on both completion AND cancellation, which is the wire by which
        // LlamaIterable.close() reaches the native cancelCompletion on real generations.
        StepVerifier.create(
                        Flux.using(
                                        () -> new TrackingIterable(tokens, closed),
                                        Flux::fromIterable,
                                        TrackingIterable::close)
                                .subscribeOn(Schedulers.boundedElastic()),
                        2)
                .expectNext(out("a"), out("b"))
                .thenRequest(2)
                .expectNext(out("c"), out("d"))
                .thenCancel()
                .verify();

        assertTrue(
                closed.get(),
                "Flux.using must call the cleanup function on cancel — this is the wire that propagates"
                        + " cancellation into llama.cpp's cancelCompletion on real generations");
    }

    /**
     * Real-model variant. Subscribes via Reactor, takes only a handful of tokens,
     * then immediately starts a second inference to verify the slot was released.
     * If cancel hadn't propagated into the native side, the second inference
     * would either block or get a busy-slot error.
     */
    @Test
    void realModel_cancelPropagatesToNativeCompletion() {
        Assumptions.assumeTrue(
                new File(TestConstants.MODEL_PATH).exists(),
                "real-model test requires " + TestConstants.MODEL_PATH);

        ModelParameters mp = new ModelParameters()
                .setModel(TestConstants.MODEL_PATH)
                .setGpuLayers(Integer.getInteger(TestConstants.PROP_TEST_NGL, 0));
        try (LlamaModel model = new LlamaModel(mp)) {
            // First: stream via Reactor with Flux.using for proper cleanup, take 3 tokens, cancel.
            String first = Flux.using(
                            () -> model.generate(
                                    new InferenceParameters("Q: 1+1=").withNPredict(20).withTemperature(0.0f)),
                            Flux::fromIterable,
                            LlamaIterable::close)
                    .subscribeOn(Schedulers.boundedElastic())
                    .take(3)
                    .map(o -> o.text)
                    .reduce("", (a, b) -> a + b)
                    .block();

            assertNotNull(first, "Reactor reduce should not produce null after take(3)");
            assertFalse(first.isEmpty(), "expected at least one token before cancel");

            // Second inference on the same model: must succeed cleanly, proving the
            // first generation's slot was released by Flux.using's cleanup function
            // routing through LlamaIterable.close() -> LlamaIterator.close() ->
            // native cancelCompletion.
            String second = model.complete(
                    new InferenceParameters("Hi").withNPredict(2).withTemperature(0.0f));
            assertNotNull(second);
        }
    }

    /** Minimal {@link LlamaOutput} for the mock test — empty probability map. */
    private static LlamaOutput out(String text) {
        return new LlamaOutput(text, Collections.<String, Float>emptyMap(), false, null);
    }

    /**
     * Test-only {@link LlamaIterable}-shaped fake: an {@code Iterable & AutoCloseable}
     * that tracks {@code close()} so the test can assert Reactor invoked it on cancel.
     * Mirrors {@link LlamaIterable}'s public contract exactly; the production class is
     * {@code final} so we can't extend it, but the {@code Iterable + AutoCloseable} pair
     * IS the contract reactive libs depend on — that is what we exercise here.
     */
    private static final class TrackingIterable implements Iterable<LlamaOutput>, AutoCloseable {
        private final List<LlamaOutput> items;
        private final AtomicBoolean closed;

        TrackingIterable(List<LlamaOutput> items, AtomicBoolean closed) {
            this.items = items;
            this.closed = closed;
        }

        @Override
        public Iterator<LlamaOutput> iterator() {
            return items.iterator();
        }

        @Override
        public void close() {
            closed.set(true);
        }
    }
}
