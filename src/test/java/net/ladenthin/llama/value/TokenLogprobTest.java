// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.value;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.empty;
import static org.hamcrest.Matchers.hasSize;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.notNullValue;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import net.ladenthin.llama.ClaudeGenerated;
import net.ladenthin.llama.json.CompletionResponseParser;
import org.junit.jupiter.api.Test;

@ClaudeGenerated(
        purpose = "Verify CompletionResponseParser.parseLogprobs populates TokenLogprob entries "
                + "including token id and nested top_logprobs/top_probs alternatives, "
                + "for both post-sampling (prob) and pre-sampling (logprob) modes.")
public class TokenLogprobTest {

    private final CompletionResponseParser parser = new CompletionResponseParser();

    @Test
    public void emptyWhenAbsent() {
        LlamaOutput out = parser.parse("{\"content\":\"hi\",\"stop\":true,\"stop_type\":\"eos\"}");
        assertThat(out.logprobs, is(empty()));
    }

    @Test
    public void parsesPostSamplingWithTopProbs() {
        String json = "{\"content\":\"hi\",\"stop\":false,"
                + "\"completion_probabilities\":["
                + "{\"token\":\"Hello\",\"id\":15043,\"prob\":0.82,"
                + "\"top_probs\":[{\"token\":\"Hi\",\"id\":9932,\"prob\":0.10},"
                + "              {\"token\":\"Hey\",\"id\":12,\"prob\":0.05}]}"
                + "]}";
        LlamaOutput out = parser.parse(json);
        assertThat(out.logprobs, hasSize(1));
        TokenLogprob first = out.logprobs.get(0);
        assertThat(first.getToken(), is("Hello"));
        assertThat(first.getTokenId(), is(15043));
        assertEquals(0.82f, first.getLogprob(), 1e-4f);
        assertThat(first.getTopLogprobs(), hasSize(2));
        assertThat(first.getTopLogprobs().get(0).getToken(), is("Hi"));
        assertThat(first.getTopLogprobs().get(0).getTokenId(), is(9932));
        assertEquals(0.10f, first.getTopLogprobs().get(0).getLogprob(), 1e-4f);
    }

    @Test
    public void parsesPreSamplingWithTopLogprobs() {
        String json = "{\"content\":\"hi\",\"stop\":false,"
                + "\"completion_probabilities\":["
                + "{\"token\":\"Hello\",\"id\":15043,\"logprob\":-0.20,"
                + "\"top_logprobs\":[{\"token\":\"Hi\",\"id\":9932,\"logprob\":-2.3}]}"
                + "]}";
        LlamaOutput out = parser.parse(json);
        assertThat(out.logprobs, hasSize(1));
        TokenLogprob first = out.logprobs.get(0);
        assertEquals(-0.20f, first.getLogprob(), 1e-4f);
        assertThat(first.getTopLogprobs(), hasSize(1));
        assertEquals(-2.3f, first.getTopLogprobs().get(0).getLogprob(), 1e-4f);
    }

    @Test
    public void preservesOrder() {
        String json = "{\"content\":\"\",\"stop\":false,"
                + "\"completion_probabilities\":["
                + "{\"token\":\"A\",\"id\":1,\"prob\":0.5},"
                + "{\"token\":\"B\",\"id\":2,\"prob\":0.4},"
                + "{\"token\":\"C\",\"id\":3,\"prob\":0.1}"
                + "]}";
        List<TokenLogprob> lp = parser.parse(json).logprobs;
        assertThat(lp.get(0).getToken(), is("A"));
        assertThat(lp.get(1).getToken(), is("B"));
        assertThat(lp.get(2).getToken(), is("C"));
    }

    @Test
    public void mapAndListBothPopulated() {
        String json = "{\"content\":\"\",\"stop\":false,"
                + "\"completion_probabilities\":["
                + "{\"token\":\"hello\",\"id\":1,\"prob\":0.9}"
                + "]}";
        LlamaOutput out = parser.parse(json);
        assertThat(out.logprobs, hasSize(1));
        assertEquals(0.9f, out.probabilities.get("hello"), 1e-4f);
    }

    @Test
    public void backwardsCompatibleConstructor() {
        LlamaOutput out =
                new LlamaOutput("hi", java.util.Collections.<String, Float>emptyMap(), false, StopReason.NONE);
        assertThat(out.logprobs, is(notNullValue()));
        assertThat(out.logprobs, is(empty()));
    }
}
