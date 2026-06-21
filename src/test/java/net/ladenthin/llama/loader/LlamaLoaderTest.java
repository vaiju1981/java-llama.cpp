// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.loader;

import static org.junit.jupiter.api.Assertions.*;

import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import net.ladenthin.llama.ClaudeGenerated;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

@ClaudeGenerated(
        purpose = "Verify the helper statics extracted from LlamaLoader without requiring any "
                + "native library: shouldCleanPath detects jllama/llama-prefixed files for "
                + "cleanup; contentsEquals performs a correct byte-level stream comparison "
                + "including BufferedInputStream wrapping and length mismatches; getTempDir "
                + "honours the 'net.ladenthin.llama.tmpdir' system-property override; and "
                + "getNativeResourcePath produces the expected classpath resource prefix; and "
                + "resourceMatchesFile compares a classpath resource to an on-disk file byte-for-byte.")
public class LlamaLoaderTest {

    private static final String TMPDIR_PROP = LlamaSystemProperties.PREFIX + ".tmpdir";

    /** A small file present on the test classpath, used as a byte-comparison fixture. */
    private static final String EXISTING_TEST_RESOURCE = "/images/test-image.jpg";

    private String previousTmpDir;

    @BeforeEach
    public void saveTmpDirProp() {
        previousTmpDir = System.getProperty(TMPDIR_PROP);
    }

    @AfterEach
    public void restoreTmpDirProp() {
        if (previousTmpDir == null) {
            System.clearProperty(TMPDIR_PROP);
        } else {
            System.setProperty(TMPDIR_PROP, previousTmpDir);
        }
    }

    // -------------------------------------------------------------------------
    // shouldCleanPath
    // -------------------------------------------------------------------------

    @Test
    public void testShouldCleanPathJllamaPrefix() {
        assertTrue(LlamaLoader.shouldCleanPath(Paths.get("/tmp/jllama.so")));
    }

    @Test
    public void testShouldCleanPathJllamaWithSuffix() {
        assertTrue(LlamaLoader.shouldCleanPath(Paths.get("/tmp/jllama-abc123.dylib")));
    }

    @Test
    public void testShouldCleanPathLlamaPrefix() {
        assertTrue(LlamaLoader.shouldCleanPath(Paths.get("/tmp/llama.dll")));
    }

    @Test
    public void testShouldCleanPathLlamaWithSuffix() {
        assertTrue(LlamaLoader.shouldCleanPath(Paths.get("/tmp/llama-model.so")));
    }

    @Test
    public void testShouldCleanPathUnrelatedFile() {
        assertFalse(LlamaLoader.shouldCleanPath(Paths.get("/tmp/somefile.so")));
    }

    @Test
    public void testShouldCleanPathEmptyFilename() {
        assertFalse(LlamaLoader.shouldCleanPath(Paths.get("/tmp/")));
    }

    @Test
    public void testShouldCleanPathPartialMatchInMiddle() {
        // "myJllama" does not start with "jllama" so should not be cleaned
        assertFalse(LlamaLoader.shouldCleanPath(Paths.get("/tmp/myjllama.so")));
    }

    @Test
    public void testShouldCleanPathCaseSensitive() {
        // "Jllama" does not start with lowercase "jllama"
        assertFalse(LlamaLoader.shouldCleanPath(Paths.get("/tmp/Jllama.so")));
    }

    // -------------------------------------------------------------------------
    // contentsEquals
    // -------------------------------------------------------------------------

    @Test
    public void testContentsEqualsIdenticalContent() throws IOException {
        byte[] data = {1, 2, 3, 4, 5};
        assertTrue(LlamaLoader.contentsEquals(new ByteArrayInputStream(data), new ByteArrayInputStream(data)));
    }

    @Test
    public void resourceMatchesFileFalseWhenResourceAbsent() throws IOException {
        java.nio.file.Path tmp = java.nio.file.Files.createTempFile("llama-loader-test", ".bin");
        try {
            java.nio.file.Files.write(tmp, new byte[] {1, 2, 3});
            // A missing classpath resource must compare as "not matching", never throw.
            assertFalse(LlamaLoader.resourceMatchesFile("/net/ladenthin/llama/does-not-exist.bin", tmp));
        } finally {
            java.nio.file.Files.deleteIfExists(tmp);
        }
    }

    @Test
    public void resourceMatchesFileTrueWhenBytesIdentical() throws IOException {
        // The fast-path reuse predicate: a present resource and a byte-identical on-disk copy match.
        java.nio.file.Path tmp = java.nio.file.Files.createTempFile("llama-loader-test", ".bin");
        try {
            try (java.io.InputStream in = LlamaLoader.class.getResourceAsStream(EXISTING_TEST_RESOURCE)) {
                assertNotNull(in, "fixture must be on the test classpath: " + EXISTING_TEST_RESOURCE);
                java.nio.file.Files.copy(in, tmp, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
            }
            assertTrue(LlamaLoader.resourceMatchesFile(EXISTING_TEST_RESOURCE, tmp));
        } finally {
            java.nio.file.Files.deleteIfExists(tmp);
        }
    }

    @Test
    public void resourceMatchesFileFalseWhenContentDiffers() throws IOException {
        // A present resource whose on-disk copy diverges (here: one extra trailing byte) must NOT match,
        // so a stale/partial file is never mistaken for the shipped library on the reuse fast path.
        java.nio.file.Path tmp = java.nio.file.Files.createTempFile("llama-loader-test", ".bin");
        try {
            try (java.io.InputStream in = LlamaLoader.class.getResourceAsStream(EXISTING_TEST_RESOURCE)) {
                assertNotNull(in, "fixture must be on the test classpath: " + EXISTING_TEST_RESOURCE);
                java.nio.file.Files.copy(in, tmp, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
            }
            java.nio.file.Files.write(tmp, new byte[] {0}, java.nio.file.StandardOpenOption.APPEND);
            assertFalse(LlamaLoader.resourceMatchesFile(EXISTING_TEST_RESOURCE, tmp));
        } finally {
            java.nio.file.Files.deleteIfExists(tmp);
        }
    }

    @Test
    public void testContentsEqualsBothEmpty() throws IOException {
        assertTrue(LlamaLoader.contentsEquals(
                new ByteArrayInputStream(new byte[0]), new ByteArrayInputStream(new byte[0])));
    }

    @Test
    public void testContentsEqualsDifferentContent() throws IOException {
        assertFalse(LlamaLoader.contentsEquals(
                new ByteArrayInputStream(new byte[] {1, 2, 3}), new ByteArrayInputStream(new byte[] {1, 2, 4})));
    }

    @Test
    public void testContentsEqualsFirstLonger() throws IOException {
        assertFalse(LlamaLoader.contentsEquals(
                new ByteArrayInputStream(new byte[] {1, 2, 3}), new ByteArrayInputStream(new byte[] {1, 2})));
    }

    @Test
    public void testContentsEqualsSecondLonger() throws IOException {
        assertFalse(LlamaLoader.contentsEquals(
                new ByteArrayInputStream(new byte[] {1, 2}), new ByteArrayInputStream(new byte[] {1, 2, 3})));
    }

    @Test
    public void testContentsEqualsAlreadyBuffered() throws IOException {
        // Passes BufferedInputStreams directly — should not double-wrap
        byte[] data = {10, 20, 30};
        assertTrue(LlamaLoader.contentsEquals(
                new BufferedInputStream(new ByteArrayInputStream(data)),
                new BufferedInputStream(new ByteArrayInputStream(data))));
    }

    @Test
    public void testContentsEqualsDifferentAtFirstByte() throws IOException {
        assertFalse(LlamaLoader.contentsEquals(
                new ByteArrayInputStream(new byte[] {0}), new ByteArrayInputStream(new byte[] {1})));
    }

    @Test
    public void testContentsEqualsSingleByteMatch() throws IOException {
        assertTrue(LlamaLoader.contentsEquals(
                new ByteArrayInputStream(new byte[] {42}), new ByteArrayInputStream(new byte[] {42})));
    }

    // -------------------------------------------------------------------------
    // getTempDir
    // -------------------------------------------------------------------------

    @Test
    public void testGetTempDirDefaultsToJavaIoTmpdir() {
        System.clearProperty(TMPDIR_PROP);
        File expected = new File(System.getProperty("java.io.tmpdir"));
        assertEquals(expected, LlamaLoader.getTempDir());
    }

    @Test
    public void testGetTempDirUsesOverrideProperty() {
        // Build path with platform separator so File.getPath() round-trips correctly
        String customPath = new File(System.getProperty("java.io.tmpdir"), "llama-test-custom").getPath();
        System.setProperty(TMPDIR_PROP, customPath);
        assertEquals(new File(customPath), LlamaLoader.getTempDir());
    }

    // -------------------------------------------------------------------------
    // getNativeResourcePath
    // -------------------------------------------------------------------------

    @Test
    public void testGetNativeResourcePathStartsWithSlash() {
        String path = LlamaLoader.getNativeResourcePath();
        assertTrue(path.startsWith("/"), "Resource path should start with '/'");
    }

    @Test
    public void testGetNativeResourcePathContainsPackage() {
        String path = LlamaLoader.getNativeResourcePath();
        // Package net.ladenthin.llama maps to net/ladenthin/llama
        assertTrue(path.contains("net/ladenthin/llama"), "Resource path should contain package");
    }

    @Test
    public void testGetNativeResourcePathContainsOsAndArch() {
        String path = LlamaLoader.getNativeResourcePath();
        // Should end with OS/arch from OSInfo
        String osArch = OSInfo.getNativeLibFolderPathForCurrentOS();
        assertTrue(path.endsWith(osArch), "Resource path should end with OS/arch: " + path);
    }

    /**
     * Regression for the layered-restructure bug: the native-library classpath
     * root is fixed at {@code /net/ladenthin/llama/<os>/<arch>} by CMakeLists +
     * the publish workflow, so it must NOT track the loader's own Java package
     * (which moved to {@code net.ladenthin.llama.loader}). Deriving it from
     * {@code LlamaLoader.class.getPackage()} produced {@code .../llama/loader/...},
     * one level too deep, so {@code getResource(...)} returned null and every
     * native-backed test failed with "No native library found".
     */
    @Test
    public void testGetNativeResourcePathIsPackageIndependent() {
        String path = LlamaLoader.getNativeResourcePath();
        String osArch = OSInfo.getNativeLibFolderPathForCurrentOS();
        assertEquals("/net/ladenthin/llama/" + osArch, path);
        assertFalse(
                path.contains("/loader/"),
                "Resource path must not include the loader subpackage — the native libs live at "
                        + "/net/ladenthin/llama/<os>/<arch>, not under the loader package: " + path);
    }
}
