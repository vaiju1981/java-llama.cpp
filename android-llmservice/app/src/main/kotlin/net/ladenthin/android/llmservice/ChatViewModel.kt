// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.android.llmservice

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import net.ladenthin.llama.LlamaModel
import net.ladenthin.llama.kotlin.generateChatFlow
import net.ladenthin.llama.parameters.InferenceParameters
import net.ladenthin.llama.parameters.ModelParameters
import net.ladenthin.llama.value.Pair

/**
 * Holds the loaded [LlamaModel] and drives streaming chat over the llama-kotlin
 * [generateChatFlow] facade. All native work runs on [Dispatchers.IO]; the UI observes
 * [uiState]. This is the whole "brain" of the app — the Compose layer is pure rendering.
 *
 * User-facing strings live in resources (localized); the UI resolves them and passes the
 * system prompt into [send], and maps [Notice] / [ErrorInfo] to localized text — so the
 * ViewModel stays free of the active locale.
 */
class ChatViewModel(application: Application) : AndroidViewModel(application) {

    /** One chat turn. [text] grows token-by-token while an assistant reply streams. */
    data class Message(val role: String, val text: String)

    /** Lifecycle of the model load. */
    enum class ModelState { NONE, LOADING, READY }

    /** One-shot user notice (mapped to a localized string + shown as a snackbar by the UI). */
    enum class Notice { SESSION_SAVED, SESSION_LOADED, NO_SESSION }

    /** Which localized error template the UI should format [ErrorInfo.detail] into. */
    enum class ErrorType { LOAD, GENERATION }

    /** A localized-at-the-UI error: the [type] selects the template, [detail] fills it in. */
    data class ErrorInfo(val type: ErrorType, val detail: String)

    /** Immutable snapshot the Compose UI renders. */
    data class UiState(
        val modelState: ModelState = ModelState.NONE,
        val modelName: String? = null,
        val messages: List<Message> = emptyList(),
        val generating: Boolean = false,
        val error: ErrorInfo? = null,
        val notice: Notice? = null,
    )

    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    private var model: LlamaModel? = null
    private var modelPath: String? = null
    private var generation: Job? = null

    /**
     * Optional chat-template override (e.g. "chatml") for GGUFs that ship no template of their
     * own. Left null for real instruct models (they carry their own template); the instrumented
     * test sets it because the tiny CI draft model has none.
     */
    private var chatTemplate: String? = null

    private companion object {
        const val MAX_TOKENS = 256
        const val CONTEXT_SIZE = 2048
    }

    /**
     * Copies a SAF-picked GGUF (a `content://` URI, which llama.cpp cannot mmap) into app-internal
     * storage, then loads it by real path. Big models take a while to copy; the UI shows LOADING.
     */
    fun loadModelFromUri(uri: Uri, displayName: String) {
        _uiState.update { it.copy(modelState = ModelState.LOADING, error = null) }
        viewModelScope.launch {
            try {
                val path = withContext(Dispatchers.IO) { copyToInternal(uri) }
                openModel(path, displayName, template = null)
            } catch (t: Throwable) {
                _uiState.update {
                    it.copy(modelState = ModelState.NONE, error = ErrorInfo(ErrorType.LOAD, t.message ?: "I/O error"))
                }
            }
        }
    }

    /**
     * Loads a model already present at an absolute on-device path (no copy). Used by the
     * instrumented test against the adb-pushed model and by session restore.
     */
    fun loadModelFromPath(path: String, template: String? = null) {
        _uiState.update { it.copy(modelState = ModelState.LOADING, error = null) }
        viewModelScope.launch {
            openModel(path, File(path).name, template)
        }
    }

    private suspend fun openModel(path: String, displayName: String, template: String?) {
        try {
            val loaded = withContext(Dispatchers.IO) {
                LlamaModel(
                    ModelParameters()
                        .setModel(path)
                        .setCtxSize(CONTEXT_SIZE)
                        .setGpuLayers(0), // CPU-only: portable across every device
                )
            }
            withContext(Dispatchers.IO) { model?.close() }
            model = loaded
            modelPath = path
            chatTemplate = template
            _uiState.update {
                it.copy(modelState = ModelState.READY, modelName = displayName, error = null)
            }
        } catch (t: Throwable) {
            _uiState.update {
                it.copy(modelState = ModelState.NONE, error = ErrorInfo(ErrorType.LOAD, t.message ?: "load failed"))
            }
        }
    }

    /**
     * Sends a user turn and streams the assistant reply into [uiState]. No-op if busy or no model.
     *
     * @param userText the message typed by the user
     * @param systemPrompt the localized system prompt (resolved from resources by the UI)
     */
    fun send(userText: String, systemPrompt: String) {
        val text = userText.trim()
        val active = model
        if (text.isEmpty() || active == null || _uiState.value.generating) {
            return
        }

        val history = _uiState.value.messages + Message("user", text)
        // Append an empty assistant message that grows as tokens arrive.
        _uiState.update { it.copy(messages = history + Message("assistant", ""), generating = true, error = null) }

        generation = viewModelScope.launch {
            val pairs = history.map { Pair(it.role, it.text) }
            var params = InferenceParameters("")
                .withMessages(systemPrompt, pairs)
                .withNPredict(MAX_TOKENS)
            chatTemplate?.let { params = params.withChatTemplate(it) }

            val reply = StringBuilder()
            try {
                active.generateChatFlow(params)
                    .flowOn(Dispatchers.IO)
                    .collect { output ->
                        reply.append(output.text)
                        _uiState.update { state -> state.replaceLastAssistant(reply.toString()) }
                    }
            } catch (t: Throwable) {
                _uiState.update { state ->
                    state.replaceLastAssistant(reply.toString())
                        .copy(error = ErrorInfo(ErrorType.GENERATION, t.message ?: "generation failed"))
                }
            } finally {
                _uiState.update { it.copy(generating = false) }
            }
        }
    }

    /** Saves the current conversation (and its model path) to private local storage. */
    fun saveSession() {
        val snapshot = _uiState.value
        val pathAtSave = modelPath
        val nameAtSave = snapshot.modelName
        val messagesAtSave = snapshot.messages
        viewModelScope.launch {
            withContext(Dispatchers.IO) {
                SessionStore.save(getApplication(), pathAtSave, nameAtSave, messagesAtSave)
            }
            _uiState.update { it.copy(notice = Notice.SESSION_SAVED) }
        }
    }

    /** Loads the saved conversation from private local storage, re-opening its model if still present. */
    fun loadSession() {
        viewModelScope.launch {
            val saved = withContext(Dispatchers.IO) { SessionStore.load(getApplication()) }
            if (saved == null) {
                _uiState.update { it.copy(notice = Notice.NO_SESSION) }
                return@launch
            }
            _uiState.update { it.copy(messages = saved.messages, notice = Notice.SESSION_LOADED) }
            val path = saved.modelPath
            if (path != null && File(path).canRead() && modelPath != path) {
                loadModelFromPath(path, chatTemplate)
            }
        }
    }

    /** Clears the one-shot [UiState.notice] after the UI has shown it. */
    fun consumeNotice() {
        _uiState.update { it.copy(notice = null) }
    }

    private fun UiState.replaceLastAssistant(newText: String): UiState {
        if (messages.isEmpty()) return this
        val updated = messages.toMutableList()
        updated[updated.lastIndex] = updated.last().copy(text = newText)
        return copy(messages = updated)
    }

    private fun copyToInternal(uri: Uri): String {
        val resolver = getApplication<Application>().contentResolver
        val target = File(getApplication<Application>().filesDir, "current-model.gguf")
        resolver.openInputStream(uri).use { input ->
            requireNotNull(input) { "content resolver returned no stream for $uri" }
            target.outputStream().use { output -> input.copyTo(output, bufferSize = 1 shl 20) }
        }
        return target.absolutePath
    }

    override fun onCleared() {
        generation?.cancel()
        // Native memory is not GC-managed. viewModelScope is already cancelled here, so close on
        // a fresh daemon thread rather than the main thread (which the native close could block).
        val toClose = model
        model = null
        if (toClose != null) {
            kotlin.concurrent.thread(isDaemon = true, name = "llama-close") { toClose.close() }
        }
    }
}
