// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;

import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link AnthropicStreamTranslator}: the OpenAI-chunk to Anthropic-SSE-event sequence
 * (message_start → text content block → tool_use blocks → message_delta → message_stop). Pure.
 */
public class AnthropicStreamTranslatorTest {

    @Test
    public void beginEmitsMessageStart() {
        AnthropicStreamTranslator translator = new AnthropicStreamTranslator("msg_1", "m");
        assertThat(translator.begin(), containsString("event: message_start"));
    }

    @Test
    public void firstTextDeltaOpensBlockThenSubsequentDeltasAppend() {
        AnthropicStreamTranslator translator = new AnthropicStreamTranslator("msg_1", "m");
        String first = translator.onChunk("{\"choices\":[{\"delta\":{\"content\":\"he\"}}]}");
        assertThat(first, containsString("event: content_block_start"));
        assertThat(first, containsString("event: content_block_delta"));
        assertThat(first, containsString("\"text\":\"he\""));
        String second = translator.onChunk("{\"choices\":[{\"delta\":{\"content\":\"llo\"}}]}");
        // No second block start; just another delta.
        assertThat(second.contains("content_block_start"), is(false));
        assertThat(second, containsString("\"text\":\"llo\""));
    }

    @Test
    public void endClosesTextBlockAndEmitsStopReasonAndMessageStop() {
        AnthropicStreamTranslator translator = new AnthropicStreamTranslator("msg_1", "m");
        translator.onChunk("{\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}");
        String end = translator.end();
        assertThat(end, containsString("event: content_block_stop"));
        assertThat(end, containsString("event: message_delta"));
        assertThat(end, containsString("\"stop_reason\":\"end_turn\""));
        assertThat(end, containsString("event: message_stop"));
    }

    @Test
    public void accumulatedToolCallsBecomeToolUseBlocksAtEnd() {
        AnthropicStreamTranslator translator = new AnthropicStreamTranslator("msg_1", "m");
        translator.onChunk("{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\","
                + "\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"Paris\\\"}\"}}]},"
                + "\"finish_reason\":\"tool_calls\"}]}");
        String end = translator.end();
        assertThat(end, containsString("event: content_block_start"));
        assertThat(end, containsString("\"type\":\"tool_use\""));
        assertThat(end, containsString("\"name\":\"get_weather\""));
        assertThat(end, containsString("event: content_block_delta"));
        assertThat(end, containsString("input_json_delta"));
        assertThat(end, containsString("\"stop_reason\":\"tool_use\""));
    }
}
