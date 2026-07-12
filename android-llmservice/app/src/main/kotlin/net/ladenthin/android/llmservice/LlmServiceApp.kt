// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.android.llmservice

import android.app.Application
import android.content.Context
import java.io.File

/**
 * Application entry point whose only job is a **privacy guarantee**: wipe the app's transient
 * working data on every **cold start**, so the app always begins fresh regardless of how it was last
 * killed (swipe-away, low-memory kill, crash — none of which reliably call `onDestroy`).
 *
 * "Working data" = the copied model ([MODEL_COPY_NAME], which can be gigabytes), the copied vision
 * projector ([MMPROJ_COPY_NAME]), and the cache dir. The **only** thing that intentionally survives
 * is the user's explicitly saved session
 * ([SessionStore]) — it exists precisely so the user can opt in to keeping a chat; everything else is
 * ephemeral. `MainActivity` additionally calls [clearWorkingData] on finish for prompt cleanup, but
 * this cold-start wipe is the reliable guarantee.
 */
class LlmServiceApp : Application() {
    override fun onCreate() {
        super.onCreate()
        clearWorkingData(this)
    }

    companion object {
        /**
         * Deletes the copied model and everything under the cache dir. Safe to call on process start
         * (nothing is loaded yet) and best-effort on exit (deleting the still-mmapped model file is
         * fine — the mapping stays valid until the model is closed). Does **not** touch the opt-in
         * saved session (`session.json`).
         *
         * @param context any context (used for `filesDir` / `cacheDir`)
         */
        fun clearWorkingData(context: Context) {
            File(context.filesDir, MODEL_COPY_NAME).delete()
            File(context.filesDir, MMPROJ_COPY_NAME).delete()
            context.cacheDir?.listFiles()?.forEach { it.deleteRecursively() }
        }
    }
}
