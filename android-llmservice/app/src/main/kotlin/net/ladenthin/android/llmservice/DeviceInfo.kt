// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.android.llmservice

import android.app.ActivityManager
import android.content.Context
import android.os.BatteryManager
import android.os.StatFs
import java.util.Locale

/**
 * A one-shot snapshot of the device's readiness for on-device inference — shown on the model-picker
 * screen so the user can gauge whether a model will fit before loading it. Read via [read]; all
 * values come from standard Android system services (no permission needed).
 *
 * @property availMemBytes currently available RAM
 * @property totalMemBytes total RAM
 * @property availStorageBytes free space on the app's data partition
 * @property batteryPercent battery level 0–100, or -1 if unknown
 * @property charging whether the device is charging
 */
data class DeviceInfo(
    val availMemBytes: Long,
    val totalMemBytes: Long,
    val availStorageBytes: Long,
    val batteryPercent: Int,
    val charging: Boolean,
) {
    companion object {
        /** Reads the current device readiness snapshot. */
        fun read(context: Context): DeviceInfo {
            val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
            val mi = ActivityManager.MemoryInfo()
            am.getMemoryInfo(mi)
            val stat = StatFs(context.filesDir.absolutePath)
            val bm = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
            val pct = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
            return DeviceInfo(
                availMemBytes = mi.availMem,
                totalMemBytes = mi.totalMem,
                availStorageBytes = stat.availableBytes,
                batteryPercent = if (pct in 0..100) pct else -1,
                charging = bm.isCharging,
            )
        }
    }
}

/** Formats a byte count as a compact human string (e.g. "3.2 GB"), locale-independent. */
fun formatBytes(bytes: Long): String {
    if (bytes < 1024) return "$bytes B"
    val units = arrayOf("KB", "MB", "GB", "TB")
    var value = bytes.toDouble() / 1024.0
    var i = 0
    while (value >= 1024.0 && i < units.size - 1) {
        value /= 1024.0
        i++
    }
    return String.format(Locale.US, "%.1f %s", value, units[i])
}
