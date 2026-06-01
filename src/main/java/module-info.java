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
 * <p>No non-implicit {@code requires} clauses are declared. The annotations from
 * JSpecify, Checker Framework qualifiers, and the Codehaus animal-sniffer annotation
 * have either {@code CLASS} retention or are not referenced from {@code module-info.java}
 * itself. The Jackson, SLF4J, and Reactive Streams API are referenced from ordinary
 * sources only; javac in the separate {@code module-info-compile} execution compiles
 * {@code module-info.java} in isolation and therefore does not need their module names.
 * Consumers that put this jar on the module path will load these dependencies through
 * their own {@code requires} graph; consumers on the classpath are unaffected.</p>
 *
 * <p>This descriptor compiles at {@code --release 9}; the rest of the source compiles
 * at {@code --release 8}. Java 8 runtimes silently ignore {@code module-info.class} at
 * the JAR root.</p>
 */
module net.ladenthin.llama {
    exports net.ladenthin.llama;
    exports net.ladenthin.llama.args;
    exports net.ladenthin.llama.json;
}
