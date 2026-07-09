// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.android.llmservice

import android.content.Context
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.app.AppCompatDelegate
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.core.os.LocaleListCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle

/**
 * Single-screen KISS demo: pick a GGUF from the file system, then chat with it fully on-device.
 * All inference is in [ChatViewModel]; this file is UI + the SAF picker + the flag language
 * picker + Save/Load session buttons.
 */
class MainActivity : AppCompatActivity() {

    val chatViewModel: ChatViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Test / automation hook: preload a model straight from an absolute path, bypassing the
        // interactive SAF picker (a separate system process the UI test can't drive). The shipping
        // UI never sets these extras.
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
        const val EXTRA_MODEL_PATH = "net.ladenthin.android.llmservice.MODEL_PATH"

        /** Optional chat-template override for template-less GGUFs (test hook). */
        const val EXTRA_CHAT_TEMPLATE = "net.ladenthin.android.llmservice.CHAT_TEMPLATE"
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ChatScreen(viewModel: ChatViewModel) {
    val state by viewModel.uiState.collectAsStateWithLifecycle()
    val context = LocalContext.current
    val snackbarHostState = remember { SnackbarHostState() }

    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        if (uri != null) {
            viewModel.loadModelFromUri(uri, queryDisplayName(context, uri))
        }
    }
    val onChoose: () -> Unit = { picker.launch(arrayOf("*/*")) }

    // One-shot notices (saved / loaded / none) -> snackbar. Resolve the localized text in
    // composable scope, then show it from the effect.
    val noticeText = state.notice?.let { stringResource(noticeRes(it)) }
    LaunchedEffect(state.notice) {
        if (noticeText != null) {
            snackbarHostState.showSnackbar(noticeText)
            viewModel.consumeNotice()
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(state.modelName ?: stringResource(R.string.app_name)) },
                actions = {
                    IconButton(
                        onClick = { viewModel.saveSession() },
                        enabled = state.messages.isNotEmpty(),
                    ) {
                        Text("💾", modifier = Modifier.testTag("saveButton"))
                    }
                    IconButton(onClick = { viewModel.loadSession() }) {
                        Text("📂", modifier = Modifier.testTag("loadButton"))
                    }
                    LanguageMenu()
                },
            )
        },
        snackbarHost = { SnackbarHost(snackbarHostState) },
    ) { padding ->
        Box(modifier = Modifier.fillMaxSize().padding(padding)) {
            val hasConversation = state.messages.isNotEmpty() || state.modelState == ChatViewModel.ModelState.READY
            when {
                state.modelState == ChatViewModel.ModelState.LOADING && !hasConversation ->
                    LoadingView()

                !hasConversation ->
                    ChooseModelView(error = errorText(state.error), onChoose = onChoose)

                else ->
                    Conversation(state = state, onSend = viewModel::send, onChoose = onChoose)
            }
        }
    }
}

@Composable
private fun LanguageMenu() {
    var expanded by remember { mutableStateOf(false) }
    val activeFlag = flagForActiveTags(AppCompatDelegate.getApplicationLocales().toLanguageTags())
    IconButton(onClick = { expanded = true }) {
        Text(activeFlag, modifier = Modifier.testTag("languageButton"))
    }
    DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
        for (language in APP_LANGUAGES) {
            DropdownMenuItem(
                text = { Text("${language.flag}  ${language.endonym}") },
                onClick = {
                    expanded = false
                    val locales =
                        if (language.tag.isEmpty()) {
                            LocaleListCompat.getEmptyLocaleList()
                        } else {
                            LocaleListCompat.forLanguageTags(language.tag)
                        }
                    // Recreates the activity in the chosen locale; the ViewModel (and the loaded
                    // model) survive the recreation, so the chat is not lost.
                    AppCompatDelegate.setApplicationLocales(locales)
                },
            )
        }
    }
}

@Composable
private fun LoadingView() {
    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        CircularProgressIndicator()
        Text(stringResource(R.string.status_loading_model), modifier = Modifier.padding(top = 16.dp))
    }
}

@Composable
private fun ChooseModelView(error: String?, onChoose: () -> Unit) {
    Column(
        modifier = Modifier.fillMaxSize().padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Text(text = stringResource(R.string.intro_pick_model), textAlign = TextAlign.Center)
        Button(
            onClick = onChoose,
            modifier = Modifier.padding(top = 24.dp).testTag("chooseModelButton"),
        ) {
            Text(stringResource(R.string.action_choose_model))
        }
        if (error != null) {
            Text(text = error, color = MaterialTheme.colorScheme.error, modifier = Modifier.padding(top = 16.dp))
        }
    }
}

@Composable
private fun Conversation(
    state: ChatViewModel.UiState,
    onSend: (String, String) -> Unit,
    onChoose: () -> Unit,
) {
    val systemPrompt = stringResource(R.string.system_prompt)
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

        if (state.modelState != ChatViewModel.ModelState.READY) {
            NotReadyBanner(state.modelState, onChoose)
        }

        errorText(state.error)?.let { message ->
            Text(
                text = message,
                color = MaterialTheme.colorScheme.error,
                modifier = Modifier.padding(horizontal = 16.dp),
            )
        }

        val ready = state.modelState == ChatViewModel.ModelState.READY
        var draft by remember { mutableStateOf("") }
        Row(
            modifier = Modifier.fillMaxWidth().padding(12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            OutlinedTextField(
                value = draft,
                onValueChange = { draft = it },
                modifier = Modifier.weight(1f).testTag("promptInput"),
                placeholder = { Text(stringResource(R.string.hint_message)) },
                enabled = ready && !state.generating,
            )
            Button(
                onClick = {
                    onSend(draft, systemPrompt)
                    draft = ""
                },
                enabled = ready && !state.generating && draft.isNotBlank(),
                modifier = Modifier.padding(start = 8.dp).testTag("sendButton"),
            ) {
                Text(stringResource(R.string.action_send))
            }
        }
    }
}

@Composable
private fun NotReadyBanner(modelState: ChatViewModel.ModelState, onChoose: () -> Unit) {
    Row(
        modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        if (modelState == ChatViewModel.ModelState.LOADING) {
            CircularProgressIndicator(modifier = Modifier.size(18.dp))
            Text(
                stringResource(R.string.status_loading_model),
                modifier = Modifier.padding(start = 12.dp),
            )
        } else {
            Text(stringResource(R.string.banner_model_not_loaded), modifier = Modifier.weight(1f))
            TextButton(onClick = onChoose) { Text(stringResource(R.string.action_choose_model)) }
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
            Text(text = message.text.ifEmpty { "…" }, modifier = Modifier.padding(12.dp))
        }
    }
}

@Composable
private fun errorText(error: ChatViewModel.ErrorInfo?): String? {
    if (error == null) return null
    val template =
        when (error.type) {
            ChatViewModel.ErrorType.LOAD -> R.string.error_load_model
            ChatViewModel.ErrorType.GENERATION -> R.string.error_generation
        }
    return stringResource(template, error.detail)
}

private fun noticeRes(notice: ChatViewModel.Notice): Int =
    when (notice) {
        ChatViewModel.Notice.SESSION_SAVED -> R.string.toast_session_saved
        ChatViewModel.Notice.SESSION_LOADED -> R.string.toast_session_loaded
        ChatViewModel.Notice.NO_SESSION -> R.string.toast_session_none
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
