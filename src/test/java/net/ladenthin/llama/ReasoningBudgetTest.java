package net.ladenthin.llama;

import java.io.File;
import java.util.Collections;

import net.ladenthin.llama.args.ReasoningFormat;
import net.ladenthin.llama.json.ChatResponseParser;
import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.Assume;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * Integration tests verifying that {@link InferenceParameters#setReasoningBudgetTokens(int)}
 * is actually enforced by the llama.cpp sampling layer when running a thinking-capable model.
 *
 * <p>These tests require the Qwen3-0.6B-Q4_K_M model (downloaded by CI). When the model file
 * is absent the entire class is skipped (same pattern as all other model-dependent test classes).
 *
 * <p>Background: a user reported that {@code setReasoningBudgetTokens()} appeared to have no
 * effect on Qwen 3.0 0.6B / 3.5 0.8B. Possible root causes are:
 * <ol>
 *   <li>The model was not entering thinking mode (missing {@code enable_thinking=true} kwarg).</li>
 *   <li>{@code reasoning_format} was not configured so thinking tokens were inline, not extracted.</li>
 *   <li>The budget mechanism in llama.cpp does not work for this model family.</li>
 * </ol>
 *
 * <p>Test 1 ({@link #testReasoningBudgetZero_suppressesThinking}) is the critical regression
 * guard: with {@code reasoning_budget_tokens=0} and thinking explicitly enabled, the sampler
 * must force-close the thinking block immediately, producing an empty {@code reasoning_content}.
 * If this test fails, the budget parameter is being ignored.
 */
@ClaudeGenerated(
        purpose = "Integration tests for setReasoningBudgetTokens() enforcement: verifies that " +
                  "budget=0 suppresses thinking tokens, budget=-1 allows them, and that thinking " +
                  "is absent when enable_thinking is not set."
)
public class ReasoningBudgetTest {

    private static LlamaModel model;
    private final ChatResponseParser parser = new ChatResponseParser();

    @BeforeClass
    public static void setup() {
        Assume.assumeTrue("Reasoning model not found, skipping ReasoningBudgetTest",
                new File(TestConstants.REASONING_MODEL_PATH).exists());
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        model = new LlamaModel(
                new ModelParameters()
                        .setModel(TestConstants.REASONING_MODEL_PATH)
                        .setCtxSize(1024)
                        .setGpuLayers(gpuLayers)
                        .setFit(false)
                        .setReasoningFormat(ReasoningFormat.DEEPSEEK)
                        .enableLogTimestamps().enableLogPrefix()
        );
    }

    @AfterClass
    public static void tearDown() {
        if (model != null) {
            model.close();
        }
    }

    /**
     * With {@code reasoning_budget_tokens=0} the sampler must force-close the thinking block
     * immediately after it opens, so {@code reasoning_content} must be empty.
     *
     * <p>This is the critical test: if it fails, the budget parameter is being silently ignored
     * by llama.cpp's sampling layer for Qwen3 models.
     */
    @Test
    public void testReasoningBudgetZero_suppressesThinking() {
        InferenceParameters params = new InferenceParameters("")
                .setMessages(null, Collections.singletonList(new Pair<>("user", "What is 2+2?")))
                .setChatTemplateKwargs(Collections.singletonMap("enable_thinking", "true"))
                .setReasoningBudgetTokens(0)
                .setNPredict(200);

        String json = model.chatComplete(params);
        String reasoningContent = parser.extractChoiceReasoningContent(json);

        Assert.assertTrue(
                "reasoning_content must be empty when reasoning_budget_tokens=0, got: " + reasoningContent,
                reasoningContent == null || reasoningContent.trim().isEmpty()
        );
    }

    /**
     * With {@code reasoning_budget_tokens=-1} (unlimited) and thinking enabled the call must
     * complete without error and produce a non-empty response. We do not assert that thinking
     * tokens are present because a small model may answer directly even when thinking is enabled.
     */
    @Test
    public void testReasoningBudgetUnlimited_completesSuccessfully() {
        InferenceParameters params = new InferenceParameters("")
                .setMessages(null, Collections.singletonList(new Pair<>("user", "What is 2+2?")))
                .setChatTemplateKwargs(Collections.singletonMap("enable_thinking", "true"))
                .setReasoningBudgetTokens(-1)
                .setNPredict(200);

        String json = model.chatComplete(params);
        Assert.assertNotNull("Response JSON must not be null", json);
        String content = parser.extractChoiceContent(json);
        Assert.assertFalse("Response content must not be empty",
                content == null || content.trim().isEmpty());
    }

    /**
     * Without {@code enable_thinking=true} in chat template kwargs, Qwen3 should not emit
     * thinking tokens. {@code reasoning_content} must be absent regardless of budget.
     */
    @Test
    public void testThinkingNotEnabled_reasoningContentAbsent() {
        InferenceParameters params = new InferenceParameters("")
                .setMessages(null, Collections.singletonList(new Pair<>("user", "What is 2+2?")))
                .setReasoningBudgetTokens(-1)
                .setNPredict(100);

        String json = model.chatComplete(params);
        String reasoningContent = parser.extractChoiceReasoningContent(json);

        Assert.assertTrue(
                "reasoning_content should be absent when thinking is not enabled, got: " + reasoningContent,
                reasoningContent == null || reasoningContent.trim().isEmpty()
        );
    }

    /**
     * With a non-zero budget, generation must complete and produce a usable answer. If reasoning
     * content is present, its length must be consistent with a 100-token budget (roughly 400–600
     * characters for typical BPE tokenisation; 800 is a generous upper bound).
     */
    @Test
    public void testReasoningBudgetLimited_doesNotExceedBudget() {
        InferenceParameters params = new InferenceParameters("")
                .setMessages(null, Collections.singletonList(
                        new Pair<>("user", "Think step by step: what is 3 times 7?")))
                .setChatTemplateKwargs(Collections.singletonMap("enable_thinking", "true"))
                .setReasoningBudgetTokens(100)
                .setNPredict(400);

        String json = model.chatComplete(params);
        String reasoningContent = parser.extractChoiceReasoningContent(json);
        String content = parser.extractChoiceContent(json);

        Assert.assertFalse("Response content must not be empty",
                content == null || content.trim().isEmpty());

        if (reasoningContent != null && !reasoningContent.trim().isEmpty()) {
            // 100 tokens * ~4–6 chars/token = 400–600 chars; 800 is a generous upper bound
            Assert.assertTrue(
                    "Reasoning content length suggests budget was exceeded (length=" +
                            reasoningContent.length() + ")",
                    reasoningContent.length() <= 800
            );
        }
    }
}
