# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT

# Intentionally (almost) empty. The net.ladenthin:llama-android AAR ships its own
# consumer ProGuard rules (proguard.txt) that keep the whole net.ladenthin.llama.**
# JNI FindClass / Jackson-reflection surface, and AndroidX / Compose / coroutines all
# ship their own consumer rules. So a minified release build needs no app-side keeps.
# Add app-specific -keep rules here if you introduce reflection in your own code.
