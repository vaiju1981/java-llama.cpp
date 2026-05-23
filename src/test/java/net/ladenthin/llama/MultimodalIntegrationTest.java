// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import java.io.File;
import java.nio.file.Paths;
import java.util.Collections;

import org.junit.AfterClass;
import org.junit.Assume;
import org.junit.BeforeClass;
import org.junit.Test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * End-to-end multimodal regression. Loads a vision-capable model + matching
 * mmproj, sends a real {@link ContentPart#imageFile(java.nio.file.Path)}
 * alongside a text prompt via the typed {@link ChatMessage#userMultimodal}
 * surface, and asserts that the model returns a non-empty reply.
 *
 * <p>The test exercises:</p>
 * <ul>
 *   <li>{@link ModelParameters#setMmproj(String)} wiring through to the native side;</li>
 *   <li>{@link InferenceParameters#setMessages(java.util.List)} serialising
 *       multipart {@code content} as the OAI array-form;</li>
 *   <li>the upstream {@code oaicompat_chat_params_parse} routing
 *       {@code image_url} blocks through the compiled-in {@code mtmd} pipeline.</li>
 * </ul>
 *
 * <p>Self-skips when any of the three system properties is unset or its file
 * is missing, so it runs only in CI (or local dev where the user staged the
 * files explicitly). See {@code .github/workflows/publish.yml} env vars
 * {@code VISION_MODEL_URL} / {@code VISION_MMPROJ_URL} / {@code VISION_IMAGE_URL}.</p>
 *
 * <p><b>Image source:</b> the CI default is
 * <a href="https://commons.wikimedia.org/wiki/File:20200601_135745_Flowers_and_Bees.jpg">
 * {@code File:20200601_135745_Flowers_and_Bees.jpg}</a> on Wikimedia Commons
 * by Bernard Ladenthin (the project copyright holder); published there under
 * CC-BY-4.0 and additionally granted under MIT to this project by the same
 * author. Any image the test machine can reach works at runtime &#x2014; the
 * URL is just an env var.</p>
 *
 * <p>Closes issues #103 and #34.</p>
 */
@ClaudeGenerated(
        purpose = "End-to-end vision regression: real vision GGUF + mmproj + author-licensed (MIT) "
                + "test image fed through the typed ChatMessage(role, List<ContentPart>) API; "
                + "asserts non-empty reply to prove the OAI multipart content round-trips through "
                + "the upstream mtmd pipeline. Closes #103 / #34."
)
public class MultimodalIntegrationTest {

    private static String modelPath;
    private static String mmprojPath;
    private static String imagePath;
    private static LlamaModel model;

    @BeforeClass
    public static void setup() {
        modelPath  = System.getProperty(TestConstants.PROP_VISION_MODEL_PATH);
        mmprojPath = System.getProperty(TestConstants.PROP_VISION_MMPROJ_PATH);
        imagePath  = System.getProperty(TestConstants.PROP_VISION_IMAGE_PATH);

        Assume.assumeTrue(
                "Vision model path not set (-D" + TestConstants.PROP_VISION_MODEL_PATH + "=...)",
                modelPath != null && !modelPath.isEmpty());
        Assume.assumeTrue(
                "Vision mmproj path not set (-D" + TestConstants.PROP_VISION_MMPROJ_PATH + "=...)",
                mmprojPath != null && !mmprojPath.isEmpty());
        Assume.assumeTrue(
                "Vision image path not set (-D" + TestConstants.PROP_VISION_IMAGE_PATH + "=...)",
                imagePath != null && !imagePath.isEmpty());

        Assume.assumeTrue("Vision model file missing: " + modelPath,
                new File(modelPath).exists());
        Assume.assumeTrue("Vision mmproj file missing: " + mmprojPath,
                new File(mmprojPath).exists());
        Assume.assumeTrue("Vision image file missing: " + imagePath,
                new File(imagePath).exists());

        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);

        model = new LlamaModel(
                new ModelParameters()
                        .setCtxSize(2048)
                        .setModel(modelPath)
                        .setMmproj(mmprojPath)
                        .setGpuLayers(gpuLayers)
                        .setFit(false));
    }

    @AfterClass
    public static void tearDown() {
        if (model != null) {
            model.close();
        }
    }

    /**
     * Build a multimodal user message, send it, and assert the reply is
     * non-empty. We do <em>not</em> assert reply content semantics &#x2014;
     * a 500M vision model can caption an apple inaccurately, and CI must
     * not flap on model quality. The point of the test is that the
     * end-to-end image-bytes &#x2192; base64 data URI &#x2192; OAI multipart
     * JSON &#x2192; upstream mtmd &#x2192; non-empty token stream pipeline
     * works without crashing.
     */
    @Test(timeout = 240_000)
    public void multimodalRequestProducesNonEmptyReply() throws Exception {
        ChatMessage userMsg = ChatMessage.userMultimodal(
                ContentPart.text("Describe what you see in this image in one short sentence."),
                ContentPart.imageFile(Paths.get(imagePath)));

        String reply = model.chatCompleteText(new InferenceParameters("")
                .setMessages(Collections.singletonList(userMsg))
                .setNPredict(48)
                .setTemperature(0.0f));

        assertNotNull("chatCompleteText must return a string, not null", reply);
        assertFalse("reply must be non-empty for a multimodal prompt; got: \"" + reply + "\"",
                reply.trim().isEmpty());
    }

    /**
     * Sanity check that a multimodal call followed by a plain text-only call
     * on the same model instance both succeed &#x2014; verifies the parts /
     * legacy split in {@code ParameterJsonSerializer.buildMessages} doesn't
     * poison the inference context for subsequent requests.
     */
    @Test(timeout = 240_000)
    public void multimodalThenTextOnSameModel() throws Exception {
        ChatMessage img = ChatMessage.userMultimodal(
                ContentPart.text("What is this?"),
                ContentPart.imageFile(Paths.get(imagePath)));
        String firstReply = model.chatCompleteText(new InferenceParameters("")
                .setMessages(Collections.singletonList(img))
                .setNPredict(24)
                .setTemperature(0.0f));
        assertNotNull(firstReply);

        ChatMessage textOnly = new ChatMessage("user", "Reply with the single word: ok");
        String secondReply = model.chatCompleteText(new InferenceParameters("")
                .setMessages(Collections.singletonList(textOnly))
                .setNPredict(8)
                .setTemperature(0.0f));
        assertNotNull(secondReply);
        assertTrue("text-only call after multimodal must still produce tokens; got: \"" + secondReply + "\"",
                secondReply.trim().length() > 0);
    }
}
