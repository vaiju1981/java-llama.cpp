// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.kotlin

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import net.ladenthin.llama.LlamaModel
import net.ladenthin.llama.parameters.InferenceParameters
import net.ladenthin.llama.value.LlamaOutput

/**
 * Streams a raw completion as a cold [Flow] of tokens.
 *
 * Each collection starts a fresh generation via [LlamaModel.generate]. The underlying
 * [net.ladenthin.llama.LlamaIterable] is closed when the flow completes **or is cancelled**, so an
 * early `take(n)`/cancellation releases the native task slot instead of leaking it.
 *
 * The token iteration blocks the collecting dispatcher; collect on a background one:
 * `model.generateFlow(params).flowOn(Dispatchers.IO)`.
 */
fun LlamaModel.generateFlow(parameters: InferenceParameters): Flow<LlamaOutput> =
    closeableIterableFlow { generate(parameters) }

/**
 * Streams an OpenAI-style chat completion as a cold [Flow] of tokens.
 *
 * Same contract as [generateFlow], backed by [LlamaModel.generateChat] (the model's chat template
 * is applied to the `messages` in [parameters]).
 */
fun LlamaModel.generateChatFlow(parameters: InferenceParameters): Flow<LlamaOutput> =
    closeableIterableFlow { generateChat(parameters) }

/**
 * Bridges a close-on-abandon iterable (the shape of `LlamaIterable`: `Iterable & AutoCloseable`)
 * into a cold [Flow]: [open] runs per collection, items are emitted in order, and the source is
 * closed on completion, error, and cancellation alike.
 *
 * Internal seam so the flow semantics are unit-testable without a loaded model
 * (see `CloseableIterableFlowTest`).
 */
internal fun <T, I> closeableIterableFlow(open: () -> I): Flow<T> where I : Iterable<T>, I : AutoCloseable =
    flow {
        open().use { source ->
            for (item in source) {
                emit(item)
            }
        }
    }
