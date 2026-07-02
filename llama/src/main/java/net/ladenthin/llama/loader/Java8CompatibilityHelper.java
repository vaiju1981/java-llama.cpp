// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
package net.ladenthin.llama.loader;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import lombok.ToString;

/**
 * Wrapper methods for Java 9+ APIs to provide Java 1.8 compatibility.
 * This class centralizes all compatibility layer logic and can be mocked for testing.
 *
 * <p>Mirrors the pattern used by the sister repo's
 * {@code net.ladenthin.maven.llamacpp.aiindex.Java8CompatibilityHelper}: each consuming
 * class declares an instance field
 * {@code private final Java8CompatibilityHelper compatibilityHelper = new Java8CompatibilityHelper();}
 * and routes Java 9+ idioms through it. The build's {@code --release 8} compiler arg
 * (see {@code pom.xml}) prevents accidental direct use of post-8 APIs in production code.
 *
 * <p>The stateless instance has no fields, so the Lombok-generated {@code toString}
 * renders as "{@code Java8CompatibilityHelper()}" — informative enough to satisfy the
 * fb-contrib IMC_IMMATURE_CLASS_NO_TOSTRING contract. Note this class also exposes a
 * {@code toString(ByteArrayOutputStream, Charset)} <em>method</em> for stream decoding;
 * that is unrelated to the generated {@link Object#toString()} override.
 */
@ToString
public class Java8CompatibilityHelper {

    /** Creates a new {@link Java8CompatibilityHelper}. */
    public Java8CompatibilityHelper() {
        // no-op
    }

    /**
     * Wrapper for {@code String#isBlank()} (Java 11+).
     * Returns {@code true} if the string is empty or contains only whitespace,
     * {@code false} otherwise.
     *
     * @param str the string to check; must not be {@code null}
     * @return {@code true} if the string is empty or blank, {@code false} otherwise
     * @throws NullPointerException if {@code str} is {@code null}
     */
    public boolean isBlank(final String str) {
        return str.isEmpty() || str.trim().isEmpty();
    }

    /**
     * Wrapper for {@code String#formatted(Object...)} (Java 15+).
     * Equivalent to {@link String#format(String, Object...)}.
     *
     * @param format the format string
     * @param args   the arguments referenced by the format specifiers in the format string
     * @return a formatted string
     */
    // Not annotated @FormatMethod because callers may pass a runtime format string;
    // marking this @FormatMethod would propagate FormatStringAnnotation to every caller.
    @SuppressWarnings("AnnotateFormatMethod")
    public String formatted(final String format, final Object... args) {
        return String.format(format, args);
    }

    /**
     * Wrapper for {@code Files#readString(Path)} (Java 11+).
     * Reads all bytes from a file and decodes them using UTF-8.
     *
     * @param path the path to the file to read
     * @return the file content as a string
     * @throws IOException if an I/O error occurs reading from the file
     */
    public String readString(final Path path) throws IOException {
        return new String(Files.readAllBytes(path), StandardCharsets.UTF_8);
    }

    /**
     * Wrapper for {@code Files#writeString(Path, CharSequence, Charset)} (Java 11+).
     * Writes a string to a file using the specified charset.
     *
     * @param path    the path to the file to write
     * @param content the string content to write
     * @param charset the charset to encode the content with; defaults to UTF-8 if {@code null}
     * @throws IOException if an I/O error occurs writing to the file
     */
    public void writeString(
            final Path path, final String content, final @org.jspecify.annotations.Nullable Charset charset)
            throws IOException {
        final Charset targetCharset = charset != null ? charset : StandardCharsets.UTF_8;
        Files.write(path, content.getBytes(targetCharset));
    }

    /**
     * Wrapper for {@code Stream#toList()} (Java 16+).
     * Collects stream elements into a {@link List}.
     *
     * @param stream the stream to collect
     * @param <T>    the element type
     * @return a list containing the stream elements
     */
    public <T> List<T> toList(final Stream<T> stream) {
        return stream.collect(Collectors.toList());
    }

    /**
     * Wrapper for {@code List#of(Object...)} (Java 9+).
     * Creates a list containing the specified elements.
     *
     * @param elements the elements to include in the list
     * @param <T>      the element type
     * @return a list containing the specified elements
     */
    // @SafeVarargs suppresses the warning at the listOf declaration; @SuppressWarnings
    // is additionally needed because javac still flags the forwarded Arrays.asList(...)
    // call as a possible-heap-pollution site even though Arrays.asList is itself
    // @SafeVarargs in the JDK.
    @SafeVarargs
    @SuppressWarnings({"unchecked", "varargs"})
    public final <T> List<T> listOf(final T... elements) {
        return Arrays.asList(elements);
    }

    // Intentionally NOT wrapped:
    // - Optional.isEmpty() (Java 11+) — use !opt.isPresent() inline instead. NullAway's
    //   CheckOptionalEmptiness recognises Optional.isPresent() / isEmpty() directly as
    //   null-narrowing for a subsequent .get(); a helper method call breaks that flow
    //   analysis. The two extra characters of !opt.isPresent() are worth the safety.
    // - Optional.orElseThrow() no-arg (Java 10+) — use orElseThrow(() -> new ...) with
    //   an explicit exception type and message at each call site. A generic wrapper
    //   would lose the per-site context that makes the failure debuggable.

    /**
     * Wrapper for {@code ByteArrayOutputStream#toString(Charset)} (Java 10+).
     * Decodes the accumulated bytes with the given charset.
     *
     * @param baos    the buffer; must not be {@code null}
     * @param charset the charset to decode with; must not be {@code null}
     * @return the decoded string
     */
    public String toString(final ByteArrayOutputStream baos, final Charset charset) {
        return new String(baos.toByteArray(), charset);
    }
}
