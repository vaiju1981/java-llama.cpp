// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.example

import android.content.Context
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle

/**
 * Single-screen KISS demo: pick a GGUF from the file system, then chat with it fully
 * on-device. All inference is in [ChatViewModel]; this file is only UI + the SAF picker.
 */
class MainActivity : ComponentActivity() {

    val chatViewModel: ChatViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Test / automation hook: preload a model straight from an absolute path,
        // bypassing the interactive SAF picker (a separate system process the UI test
        // can't drive). The shipping UI never sets these extras.
        intent?.getStringExtra(EXTRA_MODEL_PATH)?.let { path ->
            chatViewModel.loadModelFromPath(path, intent?.getStringExtra(EXTRA_CHAT_TEMPLATE))
        }

        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    ChatScreen(chatViewModel)
                }
            }
        }
    }

    companion object {
        /** Absolute on-device GGUF path to preload (test hook). */
        const val EXTRA_MODEL_PATH = "net.ladenthin.llama.example.MODEL_PATH"

        /** Optional chat-template override for template-less GGUFs (test hook). */
        const val EXTRA_CHAT_TEMPLATE = "net.ladenthin.llama.example.CHAT_TEMPLATE"
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ChatScreen(viewModel: ChatViewModel) {
    val state by viewModel.uiState.collectAsStateWithLifecycle()
    val context = androidx.compose.ui.platform.LocalContext.current

    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        if (uri != null) {
            viewModel.loadModelFromUri(uri, queryDisplayName(context, uri))
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(title = { Text(state.modelName ?: "llama.cpp Android chat") })
        },
    ) { padding ->
        Box(modifier = Modifier.fillMaxSize().padding(padding)) {
            when (state.modelState) {
                ChatViewModel.ModelState.NONE ->
                    ChooseModel(
                        error = state.error,
                        onChoose = { picker.launch(arrayOf("*/*")) },
                    )

                ChatViewModel.ModelState.LOADING ->
                    Column(
                        modifier = Modifier.fillMaxSize(),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.Center,
                    ) {
                        CircularProgressIndicator()
                        Text("Loading model…", modifier = Modifier.padding(top = 16.dp))
                    }

                ChatViewModel.ModelState.READY ->
                    Conversation(
                        state = state,
                        onSend = viewModel::send,
                    )
            }
        }
    }
}

@Composable
private fun ChooseModel(error: String?, onChoose: () -> Unit) {
    Column(
        modifier = Modifier.fillMaxSize().padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Text(
            text = "Pick a GGUF model file to run a local chat. Use an instruct/chat model for best results.",
            textAlign = TextAlign.Center,
        )
        Button(
            onClick = onChoose,
            modifier = Modifier.padding(top = 24.dp).testTag("chooseModelButton"),
        ) {
            Text("Choose GGUF model")
        }
        if (error != null) {
            Text(
                text = error,
                color = MaterialTheme.colorScheme.error,
                modifier = Modifier.padding(top = 16.dp),
            )
        }
    }
}

@Composable
private fun Conversation(state: ChatViewModel.UiState, onSend: (String) -> Unit) {
    Column(modifier = Modifier.fillMaxSize()) {
        val listState = rememberLazyListState()
        LaunchedEffect(state.messages.size, state.messages.lastOrNull()?.text) {
            if (state.messages.isNotEmpty()) {
                listState.animateScrollToItem(state.messages.lastIndex)
            }
        }
        LazyColumn(
            state = listState,
            modifier = Modifier.weight(1f).fillMaxWidth().padding(horizontal = 12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            items(state.messages) { message -> MessageBubble(message) }
        }

        if (state.error != null) {
            Text(
                text = state.error,
                color = MaterialTheme.colorScheme.error,
                modifier = Modifier.padding(horizontal = 16.dp),
            )
        }

        var draft by remember { mutableStateOf("") }
        Row(
            modifier = Modifier.fillMaxWidth().padding(12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            OutlinedTextField(
                value = draft,
                onValueChange = { draft = it },
                modifier = Modifier.weight(1f).testTag("promptInput"),
                placeholder = { Text("Message…") },
                enabled = !state.generating,
            )
            Button(
                onClick = {
                    onSend(draft)
                    draft = ""
                },
                enabled = !state.generating && draft.isNotBlank(),
                modifier = Modifier.padding(start = 8.dp).testTag("sendButton"),
            ) {
                Text("Send")
            }
        }
    }
}

@Composable
private fun MessageBubble(message: ChatViewModel.Message) {
    val isUser = message.role == "user"
    val bubbleColor =
        if (isUser) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surfaceVariant
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start,
    ) {
        Surface(
            color = bubbleColor,
            shape = MaterialTheme.shapes.medium,
            modifier = Modifier.testTag(if (isUser) "userMessage" else "assistantMessage"),
        ) {
            Text(
                text = message.text.ifEmpty { "…" },
                modifier = Modifier.padding(12.dp),
            )
        }
    }
}

private fun queryDisplayName(context: Context, uri: Uri): String {
    context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor ->
        if (cursor.moveToFirst()) {
            val index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (index >= 0) {
                return cursor.getString(index)
            }
        }
    }
    return "model.gguf"
}
