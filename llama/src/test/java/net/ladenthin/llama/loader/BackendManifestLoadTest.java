// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.loader;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import net.ladenthin.llama.ClaudeGenerated;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@ClaudeGenerated(
        purpose = "Drive LlamaLoader.initialize() end-to-end against the committed backend-manifest "
                + "fixture trees (src/test/resources/net/ladenthin/llama/Linux/backendtest*/) by "
                + "redirecting the arch component via the osinfo.architecture override. backendtest/ "
                + "holds only unloadable fake libraries, exercising every failure branch (manifest "
                + "read, per-backend temp-dir extraction, extra-file handling present and missing, "
                + "clean load-failure fallback, forced-backend fail-loud, default/cpu manifest skip, "
                + "no-manifest legacy path); backendtest-ok/ adds two trivial real x86-64 ELF dummies "
                + "so the success path, the resident-extra bookkeeping, and the same-extra clash skip "
                + "execute too. Linux-only (the trees are committed under the Linux OS folder); "
                + "self-skips elsewhere.")
public class BackendManifestLoadTest {

    /** Arch-folder override selecting the committed fixture tree {@code Linux/backendtest/}. */
    private static final String FIXTURE_ARCH = "backendtest";

    private static final String ARCH_PROP = LlamaSystemProperties.PREFIX + ".osinfo.architecture";
    private static final String TMPDIR_PROP = LlamaSystemProperties.PREFIX + ".tmpdir";
    private static final String BACKEND_PROP = LlamaSystemProperties.PREFIX + ".backend";

    private String previousArch;
    private String previousTmpDir;
    private String previousBackend;

    @TempDir
    Path tempDir;

    @BeforeEach
    public void redirectLoaderToFixtures() {
        previousArch = System.getProperty(ARCH_PROP);
        previousTmpDir = System.getProperty(TMPDIR_PROP);
        previousBackend = System.getProperty(BACKEND_PROP);
        System.setProperty(ARCH_PROP, FIXTURE_ARCH);
        System.setProperty(TMPDIR_PROP, tempDir.toString());
        System.clearProperty(BACKEND_PROP);
    }

    @AfterEach
    public void restoreProperties() {
        restore(ARCH_PROP, previousArch);
        restore(TMPDIR_PROP, previousTmpDir);
        restore(BACKEND_PROP, previousBackend);
    }

    private static void restore(String key, String value) {
        if (value == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, value);
        }
    }

    private static void assumeLinuxFixtureTree() {
        // The fixture tree is committed under the Linux OS folder; the OS path component
        // cannot be overridden, so these tests are meaningful only on a Linux JVM (which is
        // where the CI coverage run executes).
        assumeTrue("Linux".equals(OSInfo.getOSName()), "backend-manifest fixtures are committed for Linux only");
    }

    private static void assumeX86_64() {
        // The loadable dummy libraries in backendtest-ok/ are real ELF shared objects
        // compiled for x86-64 Linux (see their fixture README); loading them anywhere else
        // fails for the wrong reason.
        String osArch = System.getProperty("os.arch", "");
        assumeTrue(
                "amd64".equals(osArch) || "x86_64".equals(osArch),
                "loadable dummy backend libraries are built for x86-64 only");
    }

    @Test
    public void autoModeTriesEveryManifestBackendThenFailsWithoutDefaultLibrary() {
        assumeLinuxFixtureTree();
        UnsatisfiedLinkError error = assertThrows(UnsatisfiedLinkError.class, LlamaLoader::initialize);
        // Every manifest backend must have been attempted (and failed cleanly): the final
        // error lists each backend resource path plus the default path as tried.
        String message = error.getMessage();
        String base = "/net/ladenthin/llama/Linux/" + FIXTURE_ARCH;
        assertTrue(message.contains(base + "/fakegpu"), message);
        assertTrue(message.contains(base + "/missingextra"), message);
        assertTrue(message.contains(base + "/nodir"), message);
        assertTrue(message.contains(base + "/fallbackgpu"), message);
        assertTrue(message.contains(base), message);
        // fakegpu: the extra file is extracted into the per-backend temp subdir before its
        // load fails; the main library is never reached for that backend.
        assertTrue(Files.isRegularFile(
                tempDir.resolve(LlamaLoader.BACKEND_TEMP_DIR_PREFIX + "fakegpu").resolve("libextra.so")));
        assertFalse(Files.exists(
                tempDir.resolve(LlamaLoader.BACKEND_TEMP_DIR_PREFIX + "fakegpu").resolve("libjllama.so")));
        // fallbackgpu (no extras): its main library is extracted, then fails to load.
        assertTrue(Files.isRegularFile(tempDir.resolve(LlamaLoader.BACKEND_TEMP_DIR_PREFIX + "fallbackgpu")
                .resolve("libjllama.so")));
    }

    @Test
    public void autoModeLoadsFirstWorkingBackendAndSkipsResidentClash() throws java.io.IOException {
        assumeLinuxFixtureTree();
        assumeX86_64();
        System.setProperty(ARCH_PROP, "backendtest-ok");
        // Pre-create stale extraction artifacts so cleanup()'s recursive directory branch
        // executes on this initialize() call (deletion is not asserted: cleanup is skipped
        // when an earlier test class in the same JVM already loaded a native library).
        Path stale =
                tempDir.resolve(LlamaLoader.BACKEND_TEMP_DIR_PREFIX + "stale").resolve("nested");
        Files.createDirectories(stale);
        Files.write(stale.resolve("libjllama.so"), new byte[] {1});
        // Must succeed: extrafail's real extra library loads but its fake main library
        // fails; clash declares the same extra file name and is skipped (already resident,
        // so its temp dir is never even created); workinggpu's real dummy library loads.
        LlamaLoader.initialize();
        assertTrue(Files.isRegularFile(tempDir.resolve(LlamaLoader.BACKEND_TEMP_DIR_PREFIX + "extrafail")
                .resolve("libextra.so")));
        assertFalse(Files.exists(tempDir.resolve(LlamaLoader.BACKEND_TEMP_DIR_PREFIX + "clash")));
        assertTrue(Files.isRegularFile(tempDir.resolve(LlamaLoader.BACKEND_TEMP_DIR_PREFIX + "workinggpu")
                .resolve("libjllama.so")));
    }

    @Test
    public void withoutManifestNoBackendIsAttempted() {
        assumeLinuxFixtureTree();
        System.setProperty(ARCH_PROP, "backendtest-none");
        UnsatisfiedLinkError error = assertThrows(UnsatisfiedLinkError.class, LlamaLoader::initialize);
        // No manifest resource exists for this arch folder: the loader must take the
        // unchanged legacy path with zero backend attempts.
        String message = error.getMessage();
        assertFalse(message.contains("backendtest-none/"), message);
        assertTrue(message.contains("os.arch=backendtest-none"), message);
    }

    @Test
    public void forcedBackendFailsLoudInsteadOfFallingBack() {
        assumeLinuxFixtureTree();
        System.setProperty(BACKEND_PROP, "fakegpu");
        UnsatisfiedLinkError error = assertThrows(UnsatisfiedLinkError.class, LlamaLoader::initialize);
        assertTrue(error.getMessage().contains("Forced native backend 'fakegpu'"), error.getMessage());
    }

    @Test
    public void forcedUnknownBackendFailsLoud() {
        assumeLinuxFixtureTree();
        System.setProperty(BACKEND_PROP, "does-not-exist");
        UnsatisfiedLinkError error = assertThrows(UnsatisfiedLinkError.class, LlamaLoader::initialize);
        assertTrue(error.getMessage().contains("Forced native backend 'does-not-exist'"), error.getMessage());
    }

    @Test
    public void forcedDefaultSkipsAllManifestBackends() {
        assumeLinuxFixtureTree();
        System.setProperty(BACKEND_PROP, "default");
        UnsatisfiedLinkError error = assertThrows(UnsatisfiedLinkError.class, LlamaLoader::initialize);
        // The default library is also missing from the fixture tree (its resource path is
        // then not even listed as tried), so loading still fails — but without any backend
        // attempt: no backend path may appear in the tried list.
        String message = error.getMessage();
        assertFalse(message.contains("/fakegpu"), message);
        assertFalse(message.contains("/fallbackgpu"), message);
        assertTrue(message.contains("os.arch=" + FIXTURE_ARCH), message);
    }
}
