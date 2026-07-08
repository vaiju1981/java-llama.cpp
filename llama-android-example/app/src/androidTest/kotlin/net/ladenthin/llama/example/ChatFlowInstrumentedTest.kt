// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.example

import android.content.Intent
import androidx.compose.ui.test.junit4.createEmptyComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performTextInput
import androidx.test.core.app.ActivityScenario
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import java.io.File
import org.junit.Assert.assertFalse
import org.junit.Assume.assumeTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Real end-to-end UI test of the example app on the emulator: launches [MainActivity]
 * with a preloaded model (bypassing the interactive SAF picker), types a prompt into the
 * Compose input, taps Send, and asserts a non-empty assistant reply streamed back — so it
 * exercises the Compose UI, the ChatViewModel, the llama-kotlin generateChatFlow facade,
 * and real native inference from the AAR's jni/x86_64/libjllama.so together.
 *
 * The model is adb-pushed by the build-android-example CI job; the test self-skips when it
 * is absent, so a local connectedAndroidTest against a bare emulator stays green.
 */
@RunWith(AndroidJUnit4::class)
class ChatFlowInstrumentedTest {

    @get:Rule
    val compose = createEmptyComposeRule()

    @Test
    fun typingAndSendingStreamsAnAssistantReply() {
        assumeTrue("test model not pushed to $MODEL_PATH", File(MODEL_PATH).canRead())

        val intent = Intent(ApplicationProvider.getApplicationContext(), MainActivity::class.java)
            .putExtra(MainActivity.EXTRA_MODEL_PATH, MODEL_PATH)
            // The tiny CI draft model ships no chat template; force chatml so the chat
            // path produces tokens (real instruct models carry their own template).
            .putExtra(MainActivity.EXTRA_CHAT_TEMPLATE, "chatml")

        ActivityScenario.launch<MainActivity>(intent).use { scenario ->
            lateinit var viewModel: ChatViewModel
            scenario.onActivity { viewModel = it.chatViewModel }

            compose.waitUntil(MODEL_LOAD_TIMEOUT_MS) {
                viewModel.uiState.value.modelState == ChatViewModel.ModelState.READY
            }

            compose.onNodeWithTag("promptInput").performTextInput("Hello")
            compose.onNodeWithTag("sendButton").performClick()

            compose.waitUntil(GENERATION_TIMEOUT_MS) {
                val ui = viewModel.uiState.value
                !ui.generating && ui.messages.any { it.role == "assistant" && it.text.isNotBlank() }
            }

            val reply = viewModel.uiState.value.messages.last { it.role == "assistant" }
            assertFalse("expected a non-empty assistant reply", reply.text.isBlank())
        }
    }

    private companion object {
        /** Where the CI job adb-pushes the test GGUF (world-readable in /data/local/tmp). */
        const val MODEL_PATH = "/data/local/tmp/jllama-test-model.gguf"
        const val MODEL_LOAD_TIMEOUT_MS = 90_000L
        const val GENERATION_TIMEOUT_MS = 180_000L
    }
}
