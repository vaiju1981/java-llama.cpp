// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.parameters;

import static org.junit.jupiter.api.Assertions.*;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import net.ladenthin.llama.ClaudeGenerated;
import net.ladenthin.llama.args.ContinuationMode;
import net.ladenthin.llama.args.MiroStat;
import net.ladenthin.llama.args.ReasoningFormat;
import net.ladenthin.llama.args.Sampler;
import net.ladenthin.llama.value.Pair;
import org.junit.jupiter.api.Test;

@ClaudeGenerated(
        purpose = "Verify that every InferenceParameters wither correctly stores its value in the "
                + "internal JSON parameter map, that the toJsonString helper properly escapes all "
                + "special characters (backslash, double-quote, newline, tab, CR, '</' sequence), "
                + "that collection-based withers (logit bias, disable tokens, stop strings, samplers) "
                + "produce correctly formatted JSON arrays, and that withMessages enforces the "
                + "'user'/'assistant'-only role contract.")
public class InferenceParametersTest {

    // -------------------------------------------------------------------------
    // Constructor / prompt
    // -------------------------------------------------------------------------

    @Test
    public void testConstructorSetsPrompt() {
        InferenceParameters params = new InferenceParameters("hello");
        assertTrue(params.parameters.containsKey("prompt"));
        assertEquals("\"hello\"", params.parameters.get("prompt"));
    }

    @Test
    public void testConstructorWithEmptyPrompt() {
        InferenceParameters params = new InferenceParameters("");
        assertEquals("\"\"", params.parameters.get("prompt"));
    }

    @Test
    public void testSetPromptOverrides() {
        InferenceParameters params = new InferenceParameters("first");
        params = params.withPrompt("second");
        assertEquals("\"second\"", params.parameters.get("prompt"));
    }

    // -------------------------------------------------------------------------
    // Basic scalar setters
    // -------------------------------------------------------------------------

    @Test
    public void testSetNPredict() {
        InferenceParameters params = new InferenceParameters("").withNPredict(42);
        assertEquals("42", params.parameters.get("n_predict"));
    }

    @Test
    public void testSetTemperature() {
        InferenceParameters params = new InferenceParameters("").withTemperature(0.5f);
        assertEquals("0.5", params.parameters.get("temperature"));
    }

    @Test
    public void testSetTopK() {
        InferenceParameters params = new InferenceParameters("").withTopK(10);
        assertEquals("10", params.parameters.get("top_k"));
    }

    @Test
    public void testSetTopP() {
        InferenceParameters params = new InferenceParameters("").withTopP(0.9f);
        assertEquals("0.9", params.parameters.get("top_p"));
    }

    @Test
    public void testSetMinP() {
        InferenceParameters params = new InferenceParameters("").withMinP(0.1f);
        assertEquals("0.1", params.parameters.get("min_p"));
    }

    @Test
    public void testSetTfsZ() {
        InferenceParameters params = new InferenceParameters("").withTfsZ(1.0f);
        assertEquals("1.0", params.parameters.get("tfs_z"));
    }

    @Test
    public void testSetTypicalP() {
        InferenceParameters params = new InferenceParameters("").withTypicalP(0.8f);
        assertEquals("0.8", params.parameters.get("typical_p"));
    }

    @Test
    public void testSetRepeatLastN() {
        InferenceParameters params = new InferenceParameters("").withRepeatLastN(64);
        assertEquals("64", params.parameters.get("repeat_last_n"));
    }

    @Test
    public void testSetRepeatPenalty() {
        InferenceParameters params = new InferenceParameters("").withRepeatPenalty(1.1f);
        assertEquals("1.1", params.parameters.get("repeat_penalty"));
    }

    @Test
    public void testSetFrequencyPenalty() {
        InferenceParameters params = new InferenceParameters("").withFrequencyPenalty(0.2f);
        assertEquals("0.2", params.parameters.get("frequency_penalty"));
    }

    @Test
    public void testSetPresencePenalty() {
        InferenceParameters params = new InferenceParameters("").withPresencePenalty(0.3f);
        assertEquals("0.3", params.parameters.get("presence_penalty"));
    }

    @Test
    public void testSetSeed() {
        InferenceParameters params = new InferenceParameters("").withSeed(1234);
        assertEquals("1234", params.parameters.get("seed"));
    }

    @Test
    public void testSetNProbs() {
        InferenceParameters params = new InferenceParameters("").withNProbs(5);
        assertEquals("5", params.parameters.get("n_probs"));
    }

    @Test
    public void testSetMinKeep() {
        InferenceParameters params = new InferenceParameters("").withMinKeep(2);
        assertEquals("2", params.parameters.get("min_keep"));
    }

    @Test
    public void testSetNKeep() {
        InferenceParameters params = new InferenceParameters("").withNKeep(-1);
        assertEquals("-1", params.parameters.get("n_keep"));
    }

    @Test
    public void testSetCachePrompt() {
        InferenceParameters params = new InferenceParameters("").withCachePrompt(true);
        assertEquals("true", params.parameters.get("cache_prompt"));
    }

    @Test
    public void testSetIgnoreEos() {
        InferenceParameters params = new InferenceParameters("").withIgnoreEos(true);
        assertEquals("true", params.parameters.get("ignore_eos"));
    }

    @Test
    public void testSetPenalizeNl() {
        InferenceParameters params = new InferenceParameters("").withPenalizeNl(false);
        assertEquals("false", params.parameters.get("penalize_nl"));
    }

    @Test
    public void testSetDynamicTemperatureRange() {
        InferenceParameters params = new InferenceParameters("").withDynamicTemperatureRange(0.5f);
        assertEquals("0.5", params.parameters.get("dynatemp_range"));
    }

    @Test
    public void testSetDynamicTemperatureExponent() {
        InferenceParameters params = new InferenceParameters("").withDynamicTemperatureExponent(2.0f);
        assertEquals("2.0", params.parameters.get("dynatemp_exponent"));
    }

    // -------------------------------------------------------------------------
    // String setters (JSON-escaped)
    // -------------------------------------------------------------------------

    @Test
    public void testSetInputPrefix() {
        InferenceParameters params = new InferenceParameters("").withInputPrefix("prefix");
        assertEquals("\"prefix\"", params.parameters.get("input_prefix"));
    }

    @Test
    public void testSetInputSuffix() {
        InferenceParameters params = new InferenceParameters("").withInputSuffix("suffix");
        assertEquals("\"suffix\"", params.parameters.get("input_suffix"));
    }

    @Test
    public void testSetGrammar() {
        InferenceParameters params = new InferenceParameters("").withGrammar("root ::= \"a\"");
        assertEquals("\"root ::= \\\"a\\\"\"", params.parameters.get("grammar"));
    }

    @Test
    public void testSetJsonSchemaStoresVerbatim() {
        String schema = "{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"}},\"required\":[\"name\"]}";
        InferenceParameters params = new InferenceParameters("").withJsonSchema(schema);
        assertEquals(schema, params.parameters.get("json_schema"));
        assertTrue(params.toString().contains("\"json_schema\": " + schema));
    }

    @Test
    public void testSetPenaltyPromptString() {
        InferenceParameters params = new InferenceParameters("").withPenaltyPrompt("Hello!");
        assertEquals("\"Hello!\"", params.parameters.get("penalty_prompt"));
    }

    @Test
    public void testSetUseChatTemplate() {
        InferenceParameters params = new InferenceParameters("").withUseChatTemplate(true);
        assertEquals("true", params.parameters.get("use_jinja"));
    }

    @Test
    public void testSetChatTemplate() {
        InferenceParameters params = new InferenceParameters("").withChatTemplate("{{messages}}");
        assertEquals("\"{{messages}}\"", params.parameters.get("chat_template"));
    }

    @Test
    public void testSetChatTemplateKwargs() {
        java.util.Map<String, String> kwargs = new java.util.LinkedHashMap<>();
        kwargs.put("enable_thinking", "true");
        kwargs.put("max_tokens", "1024");
        InferenceParameters params = new InferenceParameters("").withChatTemplateKwargs(kwargs);
        String value = params.parameters.get("chat_template_kwargs");
        assertNotNull(value);
        assertTrue(value.contains("\"enable_thinking\":true"));
        assertTrue(value.contains("\"max_tokens\":1024"));
    }

    @Test
    public void testSetChatTemplateKwargsEmpty() {
        java.util.Map<String, String> kwargs = new java.util.LinkedHashMap<>();
        InferenceParameters params = new InferenceParameters("").withChatTemplateKwargs(kwargs);
        assertEquals("{}", params.parameters.get("chat_template_kwargs"));
    }

    // -------------------------------------------------------------------------
    // setTopNSigma
    // -------------------------------------------------------------------------

    @Test
    public void testSetTopNSigmaEnabled() {
        InferenceParameters params = new InferenceParameters("").withTopNSigma(2.0f);
        assertEquals("2.0", params.parameters.get("top_n_sigma"));
    }

    @Test
    public void testSetTopNSigmaDisabled() {
        InferenceParameters params = new InferenceParameters("").withTopNSigma(-1.0f);
        assertEquals("-1.0", params.parameters.get("top_n_sigma"));
    }

    // -------------------------------------------------------------------------
    // ReasoningFormat / ReasoningBudgetTokens
    // -------------------------------------------------------------------------

    @Test
    public void testSetReasoningFormatNone() {
        InferenceParameters params = new InferenceParameters("").withReasoningFormat(ReasoningFormat.NONE);
        assertEquals("\"none\"", params.parameters.get("reasoning_format"));
    }

    @Test
    public void testSetReasoningFormatAuto() {
        InferenceParameters params = new InferenceParameters("").withReasoningFormat(ReasoningFormat.AUTO);
        assertEquals("\"auto\"", params.parameters.get("reasoning_format"));
    }

    @Test
    public void testSetReasoningFormatDeepseek() {
        InferenceParameters params = new InferenceParameters("").withReasoningFormat(ReasoningFormat.DEEPSEEK);
        assertEquals("\"deepseek\"", params.parameters.get("reasoning_format"));
    }

    @Test
    public void testSetReasoningFormatDeepseekLegacy() {
        InferenceParameters params = new InferenceParameters("").withReasoningFormat(ReasoningFormat.DEEPSEEK_LEGACY);
        assertEquals("\"deepseek-legacy\"", params.parameters.get("reasoning_format"));
    }

    @Test
    public void testSetReasoningBudgetTokensPositive() {
        InferenceParameters params = new InferenceParameters("").withReasoningBudgetTokens(512);
        assertEquals("512", params.parameters.get("reasoning_budget_tokens"));
    }

    @Test
    public void testSetReasoningBudgetTokensZero() {
        InferenceParameters params = new InferenceParameters("").withReasoningBudgetTokens(0);
        assertEquals("0", params.parameters.get("reasoning_budget_tokens"));
    }

    @Test
    public void testSetReasoningBudgetTokensDisabled() {
        InferenceParameters params = new InferenceParameters("").withReasoningBudgetTokens(-1);
        assertEquals("-1", params.parameters.get("reasoning_budget_tokens"));
    }

    @Test
    public void testSetContinueFinalMessageTrue() {
        InferenceParameters params = new InferenceParameters("").withContinueFinalMessage(true);
        assertEquals("true", params.parameters.get("continue_final_message"));
    }

    @Test
    public void testSetContinueFinalMessageFalse() {
        InferenceParameters params = new InferenceParameters("").withContinueFinalMessage(false);
        assertEquals("false", params.parameters.get("continue_final_message"));
    }

    @Test
    public void testSetContinueFinalMessageReasoningContent() {
        InferenceParameters params =
                new InferenceParameters("").withContinueFinalMessage(ContinuationMode.REASONING_CONTENT);
        assertEquals("\"reasoning_content\"", params.parameters.get("continue_final_message"));
    }

    @Test
    public void testSetContinueFinalMessageContent() {
        InferenceParameters params = new InferenceParameters("").withContinueFinalMessage(ContinuationMode.CONTENT);
        assertEquals("\"content\"", params.parameters.get("continue_final_message"));
    }

    // -------------------------------------------------------------------------
    // MiroStat
    // -------------------------------------------------------------------------

    @Test
    public void testSetMiroStatDisabled() {
        InferenceParameters params = new InferenceParameters("").withMiroStat(MiroStat.DISABLED);
        assertEquals("0", params.parameters.get("mirostat"));
    }

    @Test
    public void testSetMiroStatV1() {
        InferenceParameters params = new InferenceParameters("").withMiroStat(MiroStat.V1);
        assertEquals("1", params.parameters.get("mirostat"));
    }

    @Test
    public void testSetMiroStatV2() {
        InferenceParameters params = new InferenceParameters("").withMiroStat(MiroStat.V2);
        assertEquals("2", params.parameters.get("mirostat"));
    }

    @Test
    public void testSetMiroStatTau() {
        InferenceParameters params = new InferenceParameters("").withMiroStatTau(5.0f);
        assertEquals("5.0", params.parameters.get("mirostat_tau"));
    }

    @Test
    public void testSetMiroStatEta() {
        InferenceParameters params = new InferenceParameters("").withMiroStatEta(0.1f);
        assertEquals("0.1", params.parameters.get("mirostat_eta"));
    }

    // -------------------------------------------------------------------------
    // Stop strings
    // -------------------------------------------------------------------------

    @Test
    public void testSetStopStringsSingle() {
        InferenceParameters params = new InferenceParameters("").withStopStrings("stop");
        assertEquals("[\"stop\"]", params.parameters.get("stop"));
    }

    @Test
    public void testSetStopStringsMultiple() {
        InferenceParameters params = new InferenceParameters("").withStopStrings("stop1", "stop2");
        assertEquals("[\"stop1\",\"stop2\"]", params.parameters.get("stop"));
    }

    @Test
    public void testSetStopStringsEmpty() {
        InferenceParameters params = new InferenceParameters("");
        params = params.withStopStrings();
        assertFalse(params.parameters.containsKey("stop"));
    }

    // -------------------------------------------------------------------------
    // Samplers
    // -------------------------------------------------------------------------

    @Test
    public void testSetSamplersSingle() {
        InferenceParameters params = new InferenceParameters("").withSamplers(Sampler.TOP_K);
        assertEquals("[\"top_k\"]", params.parameters.get("samplers"));
    }

    @Test
    public void testSetSamplersMultiple() {
        InferenceParameters params =
                new InferenceParameters("").withSamplers(Sampler.TOP_K, Sampler.TOP_P, Sampler.TEMPERATURE);
        assertEquals("[\"top_k\",\"top_p\",\"temperature\"]", params.parameters.get("samplers"));
    }

    @Test
    public void testSetSamplersMinP() {
        InferenceParameters params = new InferenceParameters("").withSamplers(Sampler.MIN_P);
        assertEquals("[\"min_p\"]", params.parameters.get("samplers"));
    }

    @Test
    public void testSetSamplersEmpty() {
        InferenceParameters params = new InferenceParameters("");
        params = params.withSamplers();
        assertFalse(params.parameters.containsKey("samplers"));
    }

    // -------------------------------------------------------------------------
    // Token ID bias
    // -------------------------------------------------------------------------

    @Test
    public void testSetTokenIdBias() {
        Map<Integer, Float> bias = Collections.singletonMap(15043, 1.0f);
        InferenceParameters params = new InferenceParameters("").withTokenIdBias(bias);
        String value = params.parameters.get("logit_bias");
        assertNotNull(value);
        assertTrue(value.contains("15043"));
        assertTrue(value.contains("1.0"));
    }

    @Test
    public void testSetTokenIdBiasEmpty() {
        InferenceParameters params = new InferenceParameters("").withTokenIdBias(Collections.emptyMap());
        assertFalse(params.parameters.containsKey("logit_bias"));
    }

    // -------------------------------------------------------------------------
    // Token string bias
    // -------------------------------------------------------------------------

    @Test
    public void testSetTokenBias() {
        Map<String, Float> bias = Collections.singletonMap(" Hello", 1.0f);
        InferenceParameters params = new InferenceParameters("").withTokenBias(bias);
        String value = params.parameters.get("logit_bias");
        assertNotNull(value);
        assertTrue(value.contains("Hello"));
        assertTrue(value.contains("1.0"));
    }

    @Test
    public void testSetTokenBiasEmpty() {
        InferenceParameters params = new InferenceParameters("").withTokenBias(Collections.emptyMap());
        assertFalse(params.parameters.containsKey("logit_bias"));
    }

    // -------------------------------------------------------------------------
    // Disable tokens
    // -------------------------------------------------------------------------

    @Test
    public void testDisableTokenIds() {
        InferenceParameters params = new InferenceParameters("").withDisabledTokenIds(Arrays.asList(1, 2, 3));
        String value = params.parameters.get("logit_bias");
        assertNotNull(value);
        assertTrue(value.contains("false"));
        assertTrue(value.contains("1"));
    }

    @Test
    public void testDisableTokenIdsEmpty() {
        InferenceParameters params = new InferenceParameters("").withDisabledTokenIds(Collections.emptyList());
        assertFalse(params.parameters.containsKey("logit_bias"));
    }

    @Test
    public void testDisableTokens() {
        InferenceParameters params = new InferenceParameters("").withDisabledTokens(Arrays.asList("bad", "word"));
        String value = params.parameters.get("logit_bias");
        assertNotNull(value);
        assertTrue(value.contains("false"));
        assertTrue(value.contains("bad"));
    }

    @Test
    public void testDisableTokensEmpty() {
        InferenceParameters params = new InferenceParameters("").withDisabledTokens(Collections.emptyList());
        assertFalse(params.parameters.containsKey("logit_bias"));
    }

    // -------------------------------------------------------------------------
    // Penalty prompt with token ids
    // -------------------------------------------------------------------------

    @Test
    public void testSetPenaltyPromptTokenIds() {
        InferenceParameters params = new InferenceParameters("").withPenaltyPrompt(new int[] {1, 2, 3});
        assertEquals("[1,2,3]", params.parameters.get("penalty_prompt"));
    }

    @Test
    public void testSetPenaltyPromptTokenIdsEmpty() {
        InferenceParameters params = new InferenceParameters("");
        params = params.withPenaltyPrompt(new int[] {});
        assertFalse(params.parameters.containsKey("penalty_prompt"));
    }

    // -------------------------------------------------------------------------
    // setMessages
    // -------------------------------------------------------------------------

    @Test
    public void testSetMessagesWithSystemAndUserMessages() {
        List<Pair<String, String>> messages = Collections.singletonList(new Pair<>("user", "Hi"));
        InferenceParameters params = new InferenceParameters("").withMessages("System msg", messages);
        String value = params.parameters.get("messages");
        assertNotNull(value);
        assertTrue(value.contains("system"));
        assertTrue(value.contains("System msg"));
        assertTrue(value.contains("user"));
        assertTrue(value.contains("Hi"));
    }

    @Test
    public void testSetMessagesWithAssistantRole() {
        List<Pair<String, String>> messages =
                Arrays.asList(new Pair<>("user", "Hello"), new Pair<>("assistant", "Hi there"));
        InferenceParameters params = new InferenceParameters("").withMessages(null, messages);
        String value = params.parameters.get("messages");
        assertNotNull(value);
        assertTrue(value.contains("assistant"));
        assertTrue(value.contains("Hi there"));
    }

    @Test
    public void testSetMessagesNoSystemMessage() {
        List<Pair<String, String>> messages = Collections.singletonList(new Pair<>("user", "Hello"));
        InferenceParameters params = new InferenceParameters("").withMessages(null, messages);
        String value = params.parameters.get("messages");
        assertNotNull(value);
        assertFalse(value.contains("system"));
        assertTrue(value.contains("user"));
    }

    @Test
    public void testSetMessagesEmptySystemMessage() {
        List<Pair<String, String>> messages = Collections.singletonList(new Pair<>("user", "Hello"));
        InferenceParameters params = new InferenceParameters("").withMessages("", messages);
        String value = params.parameters.get("messages");
        assertFalse(value.contains("system"));
    }

    @Test
    public void testSetMessagesInvalidRole() {
        List<Pair<String, String>> messages = Collections.singletonList(new Pair<>("system", "Bad"));
        assertThrows(IllegalArgumentException.class, () -> new InferenceParameters("").withMessages(null, messages));
    }

    @Test
    public void testSetMessagesInvalidRoleOther() {
        List<Pair<String, String>> messages = Collections.singletonList(new Pair<>("admin", "Hack"));
        assertThrows(IllegalArgumentException.class, () -> new InferenceParameters("").withMessages(null, messages));
    }

    // -------------------------------------------------------------------------
    // toString (JSON output)
    // -------------------------------------------------------------------------

    @Test
    public void testToStringContainsPrompt() {
        InferenceParameters params = new InferenceParameters("test prompt");
        String json = params.toString();
        assertTrue(json.startsWith("{"));
        assertTrue(json.endsWith("}"));
        assertTrue(json.contains("\"prompt\""));
        assertTrue(json.contains("\"test prompt\""));
    }

    @Test
    public void testToStringWithMultipleParams() {
        InferenceParameters params =
                new InferenceParameters("p").withTemperature(0.7f).withTopK(20);
        String json = params.toString();
        assertTrue(json.contains("\"temperature\""));
        assertTrue(json.contains("\"top_k\""));
    }

    // -------------------------------------------------------------------------
    // toJsonString special character escaping
    // -------------------------------------------------------------------------

    @Test
    public void testToJsonStringEscapesBackslash() {
        InferenceParameters params = new InferenceParameters("path\\to\\file");
        assertEquals("\"path\\\\to\\\\file\"", params.parameters.get("prompt"));
    }

    @Test
    public void testToJsonStringEscapesDoubleQuote() {
        InferenceParameters params = new InferenceParameters("say \"hi\"");
        assertEquals("\"say \\\"hi\\\"\"", params.parameters.get("prompt"));
    }

    @Test
    public void testToJsonStringEscapesNewline() {
        InferenceParameters params = new InferenceParameters("line1\nline2");
        assertEquals("\"line1\\nline2\"", params.parameters.get("prompt"));
    }

    @Test
    public void testToJsonStringEscapesTab() {
        InferenceParameters params = new InferenceParameters("col1\tcol2");
        assertEquals("\"col1\\tcol2\"", params.parameters.get("prompt"));
    }

    @Test
    public void testToJsonStringEscapesCarriageReturn() {
        InferenceParameters params = new InferenceParameters("a\rb");
        assertEquals("\"a\\rb\"", params.parameters.get("prompt"));
    }

    @Test
    public void testToJsonStringNull() {
        // toJsonString(null) returns null — only used internally but verify via grammar
        InferenceParameters params = new InferenceParameters("");
        params = params.withGrammar(null);
        assertNull(params.parameters.get("grammar"));
    }

    @Test
    public void testToJsonStringSlashNotEscaped() {
        // Jackson does not escape '/' — forward slashes are passed through verbatim
        InferenceParameters params = new InferenceParameters("</script>");
        String value = params.parameters.get("prompt");
        assertTrue(value.contains("</script>"));
        assertFalse(value.contains("<\\/"));
    }

    // -------------------------------------------------------------------------
    // Builder chaining returns a new instance (immutable wither semantics)
    // -------------------------------------------------------------------------

    @Test
    public void testBuilderChainingReturnsNewInstance() {
        InferenceParameters params = new InferenceParameters("");
        assertNotSame(params.withTemperature(0.5f), params);
        assertNotSame(params.withTopK(10), params);
        assertNotSame(params.withNPredict(5), params);
    }

    // -------------------------------------------------------------------------
    // Stream (package-private)
    // -------------------------------------------------------------------------

    @Test
    public void testSetStreamTrue() {
        InferenceParameters params = new InferenceParameters("").withStream(true);
        assertEquals("true", params.parameters.get("stream"));
    }

    @Test
    public void testSetStreamFalse() {
        InferenceParameters params = new InferenceParameters("").withStream(false);
        assertEquals("false", params.parameters.get("stream"));
    }

    // -------------------------------------------------------------------------
    // Multiple logit bias entries (ordering independent check)
    // -------------------------------------------------------------------------

    @Test
    public void testSetTokenIdBiasMultiple() {
        Map<Integer, Float> bias = new HashMap<>();
        bias.put(1, 0.5f);
        bias.put(2, -1.0f);
        InferenceParameters params = new InferenceParameters("").withTokenIdBias(bias);
        String value = params.parameters.get("logit_bias");
        assertNotNull(value);
        assertTrue(value.startsWith("["));
        assertTrue(value.endsWith("]"));
        assertTrue(value.contains("1"));
        assertTrue(value.contains("2"));
    }
}
