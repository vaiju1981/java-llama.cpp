// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.kotlin

import java.util.concurrent.atomic.AtomicInteger
import kotlinx.coroutines.flow.take
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.test.runTest
import org.hamcrest.MatcherAssert.assertThat
import org.hamcrest.Matchers.contains
import org.hamcrest.Matchers.instanceOf
import org.hamcrest.Matchers.`is`
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Test

/**
 * Model-free tests for [closeableIterableFlow], the seam behind
 * `LlamaModel.generateFlow`/`generateChatFlow`. A fake `Iterable & AutoCloseable` stands in for
 * `LlamaIterable` (a final class), pinning the close-on-completion / close-on-cancellation /
 * close-on-error contract that keeps native task slots from leaking.
 */
class CloseableIterableFlowTest {

    private class FakeStream(
        private val items: List<String>,
        private val failAfter: Int = Int.MAX_VALUE,
    ) : Iterable<String>, AutoCloseable {
        var closed = false
            private set

        override fun iterator(): Iterator<String> = object : Iterator<String> {
            private var index = 0

            override fun hasNext(): Boolean = index < items.size

            override fun next(): String {
                check(index < failAfter) { "simulated native failure" }
                return items[index++]
            }
        }

        override fun close() {
            closed = true
        }
    }

    @Test
    fun emitsAllItemsInOrderAndClosesOnCompletion() = runTest {
        val stream = FakeStream(listOf("a", "b", "c"))

        val collected = closeableIterableFlow<String, FakeStream> { stream }.toList()

        assertThat(collected, contains("a", "b", "c"))
        assertThat(stream.closed, `is`(true))
    }

    @Test
    fun earlyCancellationClosesTheSource() = runTest {
        val stream = FakeStream(listOf("a", "b", "c"))

        val collected = closeableIterableFlow<String, FakeStream> { stream }.take(1).toList()

        // take(1) cancels the flow after the first emission; the source must still be
        // closed so the native task slot is released.
        assertThat(collected, contains("a"))
        assertThat(stream.closed, `is`(true))
    }

    @Test
    fun iterationFailureClosesTheSourceAndPropagates() = runTest {
        val stream = FakeStream(listOf("a", "b"), failAfter = 1)

        val thrown = assertThrows(IllegalStateException::class.java) {
            kotlinx.coroutines.runBlocking {
                closeableIterableFlow<String, FakeStream> { stream }.toList()
            }
        }

        assertThat(thrown, instanceOf(IllegalStateException::class.java))
        assertThat(stream.closed, `is`(true))
    }

    @Test
    fun coldFlowOpensAFreshSourcePerCollection() = runTest {
        val opens = AtomicInteger()
        val flow = closeableIterableFlow<String, FakeStream> {
            opens.incrementAndGet()
            FakeStream(listOf("x"))
        }

        flow.toList()
        flow.toList()

        assertThat(opens.get(), `is`(2))
    }
}
