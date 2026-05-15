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
 * Integration tests for thinking/reasoning mode using Qwen3-0.6B-Q4_K_M.
 *
 * <p>These tests require the Qwen3-0.6B-Q4_K_M model (downloaded by CI). The entire
 * class is skipped when the model file is absent, matching the pattern used by all
 * other model-dependent test classes.
 *
 * <h2>Confirmed behaviour (Qwen3-0.6B, llama.cpp b9151)</h2>
 * <ol>
 *   <li><b>Thinking is active by default.</b> Qwen3's built-in chat template injects
 *       {@code <think>} into the prompt before generation starts. No extra kwarg is
 *       required; the model reasons on every request.</li>
 *   <li><b>DEEPSEEK reasoning format correctly extracts thinking tokens.</b> Setting
 *       {@code --reasoning-format deepseek} at model load time causes the server to
 *       strip the {@code <think>…</think>} block from the response body and surface it
 *       in {@code reasoning_content}.</li>
 *   <li><b>{@code reasoning_budget_tokens} is NOT enforced for Qwen3.</b> This
 *       confirms the behaviour reported by users. The root cause: Qwen3 uses
 *       <em>prompt-injected</em> thinking — the chat template writes {@code <think>}
 *       into the prompt context, so generation starts already inside a thinking block.
 *       llama.cpp's reasoning-budget sampler monitors for a <em>generated</em>
 *       {@code <think>} token; since the token is already in the prompt it never
 *       triggers, and the budget counter never starts. This is a llama.cpp limitation,
 *       not a defect in parameter serialisation (which is separately verified by
 *       {@code InferenceParametersTest} and the C++ unit tests).</li>
 * </ol>
 */
@ClaudeGenerated(
        purpose = "Integration tests for Qwen3 thinking-mode extraction and reasoning_budget_tokens " +
                  "parameter acceptance. Documents the known llama.cpp limitation that budget " +
                  "enforcement does not work for prompt-injected thinking models."
)
public class ReasoningBudgetTest {

    /**
     * Generous token budget: Qwen3-0.6B spends up to ~200 tokens thinking before answering.
     * 500 is enough for thinking + a short answer on all tested platforms.
     */
    private static final int N_PREDICT = 500;

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
                        .setCtxSize(2048)
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
     * Qwen3 enters thinking mode by default. With {@code reasoning_format=deepseek} set
     * at model level, the thinking tokens must appear in {@code reasoning_content} and
     * the final answer must appear in {@code content}.
     */
    @Test
    public void testThinkingDefault_reasoningContentAndAnswerPresent() {
        InferenceParameters params = new InferenceParameters("")
                .setMessages(null, Collections.singletonList(new Pair<>("user", "What is 2+2?")))
                .setNPredict(N_PREDICT);

        String json = model.chatComplete(params);
        String reasoningContent = parser.extractChoiceReasoningContent(json);
        String content = parser.extractChoiceContent(json);

        Assert.assertFalse(
                "reasoning_content should be non-empty (Qwen3 thinks by default)",
                reasoningContent == null || reasoningContent.trim().isEmpty());
        Assert.assertFalse(
                "content must not be empty (model must produce an answer after thinking)",
                content == null || content.trim().isEmpty());
    }

    /**
     * {@code reasoning_budget_tokens=0} is accepted by the API and the response
     * completes without error.
     *
     * <p><b>Known limitation:</b> for Qwen3, the budget is <em>not</em> enforced.
     * Qwen3's chat template injects {@code <think>} into the prompt, so generation
     * begins already inside a thinking block. llama.cpp's reasoning-budget sampler
     * only monitors for a <em>generated</em> {@code <think>} token; since it is already
     * in the prompt context the sampler never fires. As a result {@code reasoning_content}
     * remains non-empty despite the zero budget. This is a llama.cpp limitation, not a
     * bug in parameter serialisation.
     */
    @Test
    public void testReasoningBudgetZero_parameterAccepted_thinkingNotSuppressed() {
        InferenceParameters params = new InferenceParameters("")
                .setMessages(null, Collections.singletonList(new Pair<>("user", "What is 2+2?")))
                .setReasoningBudgetTokens(0)
                .setNPredict(N_PREDICT);

        String json = model.chatComplete(params);

        // The call must complete without throwing.
        Assert.assertNotNull("Response JSON must not be null", json);

        // Document current (broken) behaviour: reasoning_content is non-empty even
        // though budget=0 should have suppressed it.  This assertion will start FAILING
        // once llama.cpp adds support for prompt-prefilled thinking contexts, which is
        // the signal to flip it to assertFalse and close the limitation.
        String reasoningContent = parser.extractChoiceReasoningContent(json);
        Assert.assertFalse(
                "reasoning_content is expected to be present because budget enforcement " +
                "does not work for Qwen3 (prompt-injected thinking). " +
                "If this assertion fails, budget enforcement has been fixed — update the test.",
                reasoningContent == null || reasoningContent.trim().isEmpty());
    }

    /**
     * A positive {@code reasoning_budget_tokens} value is accepted, the call completes,
     * and the model produces a non-empty answer.
     *
     * <p>See {@link #testReasoningBudgetZero_parameterAccepted_thinkingNotSuppressed} for
     * the note on why the budget count itself is not asserted.
     */
    @Test
    public void testReasoningBudgetPositive_parameterAccepted() {
        InferenceParameters params = new InferenceParameters("")
                .setMessages(null, Collections.singletonList(
                        new Pair<>("user", "Think step by step: what is 3 times 7?")))
                .setReasoningBudgetTokens(100)
                .setNPredict(N_PREDICT);

        String json = model.chatComplete(params);
        Assert.assertNotNull("Response JSON must not be null", json);
        String content = parser.extractChoiceContent(json);
        Assert.assertFalse("content must not be empty",
                content == null || content.trim().isEmpty());
    }
}
