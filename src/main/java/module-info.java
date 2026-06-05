// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

/**
 * JPMS module descriptor for the java-llama.cpp JNI bindings.
 *
 * <p>Exports the three hand-written public packages
 * ({@code net.ladenthin.llama}, {@code net.ladenthin.llama.args},
 * {@code net.ladenthin.llama.json}). The native libraries shipped under
 * {@code /net/ladenthin/llama/{OS}/{ARCH}/} are loaded by
 * {@link net.ladenthin.llama.LlamaLoader} via
 * {@link Class#getResourceAsStream(String)} on its own class object, so the resources
 * are looked up in this module and do <em>not</em> need to be {@code opens}'d.</p>
 *
 * <p>JSpecify {@code @NullMarked} is declared at the module level here so that no source
 * file compiled at {@code --release 8} references the JSpecify annotation type directly.
 * Otherwise javac would emit an unsuppressible {@code unknown enum constant
 * ElementType.MODULE} classfile-read warning for each source compiled at release 8 that
 * resolves {@code @NullMarked} ({@code @NullMarked} carries
 * {@code @Target({MODULE, PACKAGE, TYPE})} and Java 8 does not know about
 * {@code ElementType.MODULE}). Confining the reference to {@code module-info.java} —
 * which compiles at {@code --release 9} — keeps that warning out of the build entirely.</p>
 *
 * <p>{@code requires static org.jspecify} is needed only at compile time of this
 * descriptor; JSpecify annotations carry {@code RetentionPolicy.CLASS} so module-path
 * consumers never need jspecify on their runtime path. Checker Framework qualifiers and
 * the Codehaus animal-sniffer annotation are likewise compile-time only. Jackson, SLF4J,
 * and Reactive Streams API are referenced from ordinary sources only; javac in the
 * separate {@code module-info-compile} execution compiles {@code module-info.java} in
 * isolation and therefore does not need their module names. Consumers that put this jar
 * on the module path will load these dependencies through their own {@code requires}
 * graph; consumers on the classpath are unaffected.</p>
 *
 * <p>This descriptor compiles at {@code --release 9}; the rest of the source compiles
 * at {@code --release 8}. Java 8 runtimes silently ignore {@code module-info.class} at
 * the JAR root.</p>
 */
@org.jspecify.annotations.NullMarked
module net.ladenthin.llama {
    requires static org.jspecify;

    // Lombok is `provided` scope: only used at compile time to generate equals/hashCode/toString.
    // `requires static` means the runtime does not need the lombok jar on the module path —
    // the @lombok.Generated annotation carried on generated members has CLASS retention.
    requires static lombok;

    exports net.ladenthin.llama;
    exports net.ladenthin.llama.args;
    exports net.ladenthin.llama.json;
}
