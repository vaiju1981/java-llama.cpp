// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.value;

import com.fasterxml.jackson.databind.JsonNode;
import lombok.EqualsAndHashCode;

/** Typed view of one entry in llama.cpp's server-metrics {@code slots} array. */
@EqualsAndHashCode
public final class SlotMetrics {

    private final JsonNode node;

    /**
     * Wrap a raw slot metrics object.
     *
     * @param node slot JSON emitted by llama.cpp
     */
    public SlotMetrics(JsonNode node) {
        this.node = node;
    }

    /**
     * Returns the zero-based server slot identifier.
     * @return slot identifier
     */
    public int getId() {
        return node.path("id").asInt(-1);
    }

    /**
     * Returns the context capacity assigned to this slot.
     * @return context capacity
     */
    public int getContextSize() {
        return node.path("n_ctx").asInt(0);
    }

    /**
     * Reports whether this slot is currently processing a task.
     * @return {@code true} while processing
     */
    public boolean isProcessing() {
        return node.path("is_processing").asBoolean(false);
    }

    /**
     * Returns the logical prompt-token count for the current or most recent task.
     * @return logical prompt-token count
     */
    public long getPromptTokens() {
        return node.path("n_prompt_tokens").asLong(0L);
    }

    /**
     * Returns prompt tokens evaluated by the model for the current or most recent task.
     * @return evaluated prompt-token count
     */
    public long getProcessedPromptTokens() {
        return node.path("n_prompt_tokens_processed").asLong(0L);
    }

    /**
     * Returns prompt tokens reused from KV cache for the current or most recent task.
     * @return cached prompt-token count
     */
    public long getCachedPromptTokens() {
        return node.path("n_prompt_tokens_cache").asLong(0L);
    }

    /**
     * Returns tokens decoded for the current or most recent task.
     * @return decoded-token count
     */
    public long getDecodedTokens() {
        return node.path("next_token").path("n_decoded").asLong(0L);
    }

    /**
     * Returns tokens remaining under the current generation limit.
     * @return remaining-token count
     */
    public long getRemainingTokens() {
        return node.path("next_token").path("n_remain").asLong(0L);
    }

    /**
     * Returns raw slot JSON for fields not represented by typed accessors.
     * @return raw slot JSON
     */
    public JsonNode asJson() {
        return node;
    }

    @Override
    public String toString() {
        return node.toString();
    }
}
