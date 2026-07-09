// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.android.llmservice

import android.content.Context
import java.io.File
import org.json.JSONArray
import org.json.JSONObject

/**
 * Saves and loads the chat session to a single JSON file in the app's **private** internal
 * storage ({@code filesDir}) — readable only by this app, never uploaded anywhere. This keeps
 * the "nothing leaves the device" promise: save/load is opt-in (Save/Load buttons) and local.
 *
 * The file records the conversation plus the model's on-device path and display name, so a
 * loaded session can also re-open its model. Uses {@code org.json} (bundled in Android), so no
 * dependency is added.
 */
object SessionStore {

    private const val FILE_NAME = "session.json"
    private const val KEY_MODEL_PATH = "modelPath"
    private const val KEY_MODEL_NAME = "modelName"
    private const val KEY_MESSAGES = "messages"
    private const val KEY_ROLE = "role"
    private const val KEY_TEXT = "text"

    /** A restored session: the conversation plus the model it was held with (if still known). */
    data class Saved(
        val modelPath: String?,
        val modelName: String?,
        val messages: List<ChatViewModel.Message>,
    )

    /** Overwrites the single saved session with the current conversation. */
    fun save(
        context: Context,
        modelPath: String?,
        modelName: String?,
        messages: List<ChatViewModel.Message>,
    ) {
        val root = JSONObject()
        root.put(KEY_MODEL_PATH, modelPath ?: JSONObject.NULL)
        root.put(KEY_MODEL_NAME, modelName ?: JSONObject.NULL)
        val array = JSONArray()
        for (message in messages) {
            array.put(JSONObject().put(KEY_ROLE, message.role).put(KEY_TEXT, message.text))
        }
        root.put(KEY_MESSAGES, array)
        File(context.filesDir, FILE_NAME).writeText(root.toString())
    }

    /** Reads the saved session, or {@code null} if none exists or the file is unreadable/corrupt. */
    fun load(context: Context): Saved? {
        val file = File(context.filesDir, FILE_NAME)
        if (!file.exists()) {
            return null
        }
        return try {
            val root = JSONObject(file.readText())
            val array = root.optJSONArray(KEY_MESSAGES) ?: JSONArray()
            val messages = ArrayList<ChatViewModel.Message>(array.length())
            for (i in 0 until array.length()) {
                val entry = array.getJSONObject(i)
                messages.add(ChatViewModel.Message(entry.optString(KEY_ROLE), entry.optString(KEY_TEXT)))
            }
            Saved(
                modelPath = root.optStringOrNull(KEY_MODEL_PATH),
                modelName = root.optStringOrNull(KEY_MODEL_NAME),
                messages = messages,
            )
        } catch (t: Throwable) {
            null
        }
    }

    private fun JSONObject.optStringOrNull(key: String): String? =
        if (isNull(key)) null else optString(key).ifEmpty { null }
}
