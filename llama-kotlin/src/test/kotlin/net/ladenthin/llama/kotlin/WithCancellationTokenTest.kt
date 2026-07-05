// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.kotlin

import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import net.ladenthin.llama.callback.CancellationToken
import org.hamcrest.MatcherAssert.assertThat
import org.hamcrest.Matchers.`is`
import org.junit.jupiter.api.Test

/**
 * Model-free tests for [withCancellationToken], the seam behind `LlamaModel.completeSuspend`.
 * A blocking block that spins on the token stands in for the native inference loop, pinning
 * that coroutine cancellation reaches the cooperative [CancellationToken] (and that normal
 * completion does not).
 */
class WithCancellationTokenTest {

    @Test
    fun returnsBlockResultAndLeavesTokenUncancelledOnNormalCompletion() = runBlocking {
        val seenToken = AtomicReference<CancellationToken>()

        val result = withCancellationToken(Dispatchers.IO) { token ->
            seenToken.set(token)
            "done"
        }

        assertThat(result, `is`("done"))
        assertThat(seenToken.get().isCancelled, `is`(false))
    }

    @Test
    fun coroutineCancellationCancelsTheTokenSoTheBlockingLoopStops() = runBlocking {
        val blockEntered = CountDownLatch(1)
        val blockFinished = CountDownLatch(1)
        val seenToken = AtomicReference<CancellationToken>()

        // launch on Default (not runBlocking's single event-loop thread): the test
        // body blocks on latches below, which would otherwise starve the event loop
        // before the job is ever dispatched.
        val job = launch(Dispatchers.Default) {
            withCancellationToken(Dispatchers.IO) { token ->
                seenToken.set(token)
                blockEntered.countDown()
                // Stand-in for the native token loop: spins until the cooperative
                // token is cancelled. If cancellation never reaches the token this
                // spins forever and the withTimeout below fails the test.
                while (!token.isCancelled) {
                    Thread.sleep(5)
                }
                blockFinished.countDown()
                "partial"
            }
        }

        assertThat(blockEntered.await(5, TimeUnit.SECONDS), `is`(true))
        withTimeout(5_000) { job.cancelAndJoin() }

        // The block observed the cancel and returned (the scope waited for it),
        // proving no still-running generation leaks past the suspend call.
        assertThat(blockFinished.await(5, TimeUnit.SECONDS), `is`(true))
        assertThat(seenToken.get().isCancelled, `is`(true))
        assertThat(job.isCancelled, `is`(true))
    }
}
