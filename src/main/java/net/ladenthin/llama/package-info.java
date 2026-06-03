// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

/**
 * Java bindings for llama.cpp.
 *
 * <p>JSpecify {@code @NullMarked} is declared at module level in
 * {@code module-info.java} and applies transitively to every package
 * in this module: every parameter, return value, and field is non-null
 * unless explicitly annotated {@code @Nullable}. NullAway and the
 * Checker Framework Nullness Checker both enforce this at compile
 * time via the configured Error Prone compiler plugin (see
 * {@code pom.xml}). Public-API methods that may legitimately have
 * no value prefer {@code java.util.Optional<T>} over
 * {@code @Nullable T}.
 */
package net.ladenthin.llama;
