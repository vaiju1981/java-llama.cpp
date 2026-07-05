// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.kotlin

import kotlin.coroutines.CoroutineContext
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.withContext
import net.ladenthin.llama.LlamaModel
import net.ladenthin.llama.callback.CancellationToken
import net.ladenthin.llama.parameters.ChatRequest
import net.ladenthin.llama.parameters.InferenceParameters
import net.ladenthin.llama.value.ChatResponse

/**
 * Runs a blocking completion as a suspending call on [context] (default [Dispatchers.IO]),
 * with **coroutine cancellation wired to the binding's cooperative [CancellationToken]**:
 * cancelling the calling coroutine cancels the token, the native loop stops at the next token
 * boundary (freeing the slot), and the [CancellationException] propagates as usual.
 */
suspend fun LlamaModel.completeSuspend(
    parameters: InferenceParameters,
    context: CoroutineContext = Dispatchers.IO,
): String = withCancellationToken(context) { token -> complete(parameters, token) }

/**
 * Runs a typed chat completion as a suspending call on [context] (default [Dispatchers.IO]).
 *
 * Not token-cancellable: [LlamaModel.chat] has no [CancellationToken] overload, so a cancelled
 * coroutine resumes only after the native call returns. Use [completeSuspend] or
 * [generateChatFlow] when prompt cancellation matters.
 */
suspend fun LlamaModel.chatSuspend(
    request: ChatRequest,
    context: CoroutineContext = Dispatchers.IO,
): ChatResponse = withContext(context) { chat(request) }

/**
 * Runs an OpenAI-style chat completion and returns only the assistant text, as a suspending
 * call on [context] (default [Dispatchers.IO]). Same cancellation caveat as [chatSuspend].
 */
suspend fun LlamaModel.chatCompleteTextSuspend(
    parameters: InferenceParameters,
    context: CoroutineContext = Dispatchers.IO,
): String = withContext(context) { chatCompleteText(parameters) }

/**
 * Computes an embedding as a suspending call on [context] (default [Dispatchers.IO]).
 */
suspend fun LlamaModel.embedSuspend(
    prompt: String,
    context: CoroutineContext = Dispatchers.IO,
): FloatArray = withContext(context) { embed(prompt) }

/**
 * Runs [block] with a fresh [CancellationToken] on [context] and cancels that token as soon as
 * the calling coroutine is cancelled, so a cooperative native loop stops at its next check
 * instead of running to natural completion.
 *
 * Structure: the blocking [block] runs in a child [async]; awaiting it is the cancellable
 * suspension point. On cancellation the token is cancelled *before* rethrowing, and the
 * enclosing [coroutineScope] then waits for [block] to observe the token and return — so the
 * function never leaks a still-running generation past its own return.
 *
 * Internal seam so the wiring is unit-testable without a loaded model (see
 * `WithCancellationTokenTest`).
 */
internal suspend fun <R> withCancellationToken(
    context: CoroutineContext,
    block: (CancellationToken) -> R,
): R = coroutineScope {
    val token = CancellationToken()
    val work = async(context) { block(token) }
    try {
        work.await()
    } catch (e: CancellationException) {
        token.cancel()
        throw e
    }
}
