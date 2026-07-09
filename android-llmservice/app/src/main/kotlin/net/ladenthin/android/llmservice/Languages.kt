// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.android.llmservice

/**
 * One selectable UI language for the flag picker.
 *
 * @property tag BCP-47 language tag for [androidx.appcompat.app.AppCompatDelegate.setApplicationLocales]
 *   (empty string = follow the system language).
 * @property endonym the language's own name, shown the same in every locale (e.g. "Deutsch").
 * @property flag a representative flag emoji. NOTE: a flag is a country, not a language — this is
 *   a pragmatic, friendly visual only, not a claim that the language belongs to that country.
 */
data class AppLanguage(val tag: String, val endonym: String, val flag: String)

/**
 * The languages the app ships translations for. Kept in sync with the values-*/ resource dirs
 * and res/xml/locales_config.xml. The first entry (empty tag) follows the device language.
 */
val APP_LANGUAGES: List<AppLanguage> = listOf(
    AppLanguage("", "System", "🌐"),
    AppLanguage("en", "English", "🇬🇧"),
    AppLanguage("de", "Deutsch", "🇩🇪"),
    AppLanguage("es", "Español", "🇪🇸"),
    AppLanguage("fr", "Français", "🇫🇷"),
    AppLanguage("it", "Italiano", "🇮🇹"),
    AppLanguage("pt", "Português", "🇵🇹"),
    AppLanguage("ru", "Русский", "🇷🇺"),
    AppLanguage("tr", "Türkçe", "🇹🇷"),
    AppLanguage("ar", "العربية", "🇸🇦"),
    AppLanguage("hi", "हिन्दी", "🇮🇳"),
    AppLanguage("zh-CN", "中文", "🇨🇳"),
    AppLanguage("ja", "日本語", "🇯🇵"),
    AppLanguage("ko", "한국어", "🇰🇷"),
)

/**
 * Picks the flag to show in the top bar for the currently active language tags. Matches by the
 * primary language subtag (so "de-DE" still shows the German flag); falls back to the globe.
 *
 * @param activeTags comma-joined BCP-47 tags from the current app locales (may be empty)
 * @return the matching flag emoji, or the globe when nothing matches
 */
fun flagForActiveTags(activeTags: String): String {
    if (activeTags.isBlank()) return "🌐"
    val primary = activeTags.substringBefore(',').trim()
    val lang = primary.substringBefore('-').lowercase()
    return APP_LANGUAGES.firstOrNull { it.tag.equals(primary, ignoreCase = true) }?.flag
        ?: APP_LANGUAGES.firstOrNull { it.tag.isNotEmpty() && it.tag.substringBefore('-').lowercase() == lang }?.flag
        ?: "🌐"
}
