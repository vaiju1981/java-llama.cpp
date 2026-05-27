// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

/**
 * Resolves library-specific system properties under the {@link #PREFIX} domain prefix.
 */
public class LlamaSystemProperties {

    /** Creates a new {@link LlamaSystemProperties}. */
    public LlamaSystemProperties() {
    }

    /** Common system-property prefix for all library-specific overrides. */
    public static final String PREFIX = "net.ladenthin.llama";

    private String getProperty(String suffix) {
        return System.getProperty(PREFIX + suffix);
    }

    /**
     * Custom directory containing the native jllama shared library.
     *
     * @return the configured library directory, or {@code null} if unset
     */
    public String getLibPath() {
        return getProperty(".lib.path");
    }

    /**
     * Override for the native library file name.
     *
     * @return the configured library file name, or {@code null} if unset
     */
    public String getLibName() {
        return getProperty(".lib.name");
    }

    /**
     * Custom temporary directory used when extracting the native library from
     * the JAR. Falls back to {@code java.io.tmpdir} if absent.
     *
     * @return the configured temp directory, or {@code null} if unset
     */
    public String getTmpDir() {
        return getProperty(".tmpdir");
    }

    /**
     * Architecture override for OS/arch detection in {@link OSInfo}.
     *
     * @return the configured architecture override, or {@code null} if unset
     */
    public String getOsinfoArchitecture() {
        return getProperty(".osinfo.architecture");
    }

    /**
     * Number of GPU layers used in tests; parsed by the test suite.
     *
     * @return the configured GPU layer count as a string, or {@code null} if unset
     */
    public String getTestNgl() {
        return getProperty(".test.ngl");
    }
}
