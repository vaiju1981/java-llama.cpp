// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

/**
 * Java bindings for llama.cpp.
 *
 * <p>The package is JSpecify {@code @NullMarked}: every parameter, return
 * value, and field is non-null unless explicitly annotated {@code @Nullable}.
 * NullAway enforces this at compile time via the configured Error Prone
 * compiler plugin (see {@code pom.xml}). Public-API methods that may
 * legitimately have no value prefer {@code java.util.Optional<T>} over
 * {@code @Nullable T}.
 */
@NullMarked
package net.ladenthin.llama;

import org.jspecify.annotations.NullMarked;
