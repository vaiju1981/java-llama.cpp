// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.loader;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;
import lombok.ToString;
import org.jspecify.annotations.Nullable;

/**
 * Set the system property {@code net.ladenthin.llama.lib.path} appropriately
 * so that the library can find {@code *.dll}, {@code *.dylib} and
 * {@code *.so} files, according to the current OS (Windows, Linux, macOS).
 *
 * <p>The library files are automatically extracted from this project's package (JAR).
 *
 * <p>Historically the loader also honoured a {@code net.ladenthin.llama.lib.name}
 * property that overrode the resolved library filename. Upstream removed the
 * code path that read it in {@code kherud/java-llama.cpp} commit {@code 6bb63e1}
 * (&quot;add ggml shared library to binding&quot;) when the loader was extended to
 * load multiple shared libraries (ggml + jllama) as separate files &mdash; the
 * single-name-override model is incompatible with that. The Javadoc mention
 * has since been a documentation lie in both upstream and this fork; it has
 * now been removed here, and the corresponding {@code getLibName()} getter
 * has been deleted from {@code LlamaSystemProperties}.
 *
 * <p>usage: call {@link #initialize()} before using the library.
 *
 * @author leo
 */
@SuppressWarnings("UseOfSystemOutOrSystemErr")
@ToString
public class LlamaLoader {

    /**
     * Private monitor guarding {@link #initialize()}. Synchronizing on this
     * dedicated object instead of {@code LlamaLoader.class} keeps the lock
     * private to this class, so untrusted code that can reach the public
     * {@code LlamaLoader} type cannot acquire the same intrinsic lock and
     * interfere with library initialization (SpotBugs
     * {@code USO_UNSAFE_STATIC_METHOD_SYNCHRONIZATION}).
     */
    private static final Object INITIALIZE_LOCK = new Object();

    private static boolean extracted = false;
    private static final LlamaSystemProperties systemProperties = new LlamaSystemProperties();
    private static final NativeLibraryPermissionSetter permissionSetter = new NativeLibraryPermissionSetter(System.err);

    /**
     * Canonical classpath root for the bundled native libraries. Fixed by
     * {@code CMakeLists.txt} and the publish workflow (both emit to
     * {@code resources/net/ladenthin/llama/<os>/<arch>/}); it must NOT be
     * derived from this loader's own Java package, which moved to
     * {@code net.ladenthin.llama.loader} during the layered restructure.
     */
    private static final String NATIVE_RESOURCE_BASE = "/net/ladenthin/llama";

    /** Static utility holder; not instantiable. */
    private LlamaLoader() {}

    /**
     * Loads the llama and jllama shared libraries
     */
    public static void initialize() {
        synchronized (INITIALIZE_LOCK) {
            // only cleanup before the first extract
            if (!extracted) {
                cleanup();
            }
            if ("Mac".equals(OSInfo.getOSName())) {
                String nativeDirName = getNativeResourcePath();
                String tempFolder = getTempDir().getAbsolutePath();
                System.out.println(nativeDirName);
                Path metalFilePath = extractFile(nativeDirName, "ggml-metal.metal", tempFolder);
                if (metalFilePath == null) {
                    System.err.println("'ggml-metal.metal' not found");
                }
            }
            loadNativeLibrary("jllama");
            extracted = true;
        }
    }

    /**
     * Deleted old native libraries e.g. on Windows the DLL file is not removed on VM-Exit (bug #80)
     */
    private static void cleanup() {
        try (Stream<Path> dirList = Files.list(getTempDir().toPath())) {
            dirList.filter(LlamaLoader::shouldCleanPath).forEach(LlamaLoader::cleanPath);
        } catch (IOException e) {
            System.err.println("Failed to open directory: " + e.getMessage());
        }
    }

    static boolean shouldCleanPath(Path path) {
        Path fileNamePath = path.getFileName();
        if (fileNamePath == null) {
            return false;
        }
        String fileName = fileNamePath.toString();
        return fileName.startsWith("jllama") || fileName.startsWith("llama");
    }

    private static void cleanPath(Path path) {
        try {
            Files.delete(path);
        } catch (Exception e) {
            System.err.println("Failed to delete old native lib: " + e.getMessage());
        }
    }

    private static void loadNativeLibrary(String name) {
        List<String> triedPaths = new ArrayList<>();

        String nativeLibName = System.mapLibraryName(name);
        String nativeLibPath = systemProperties.getLibPath();
        if (nativeLibPath != null) {
            Path path = Paths.get(nativeLibPath, nativeLibName);
            if (loadNativeLibrary(path)) {
                return;
            } else {
                triedPaths.add(nativeLibPath);
            }
        }

        if (OSInfo.isAndroid()) {
            try {
                // loadLibrary can load directly from packed apk file automatically
                // if java-llama.cpp is added as code source
                System.loadLibrary(name);
                return;
            } catch (UnsatisfiedLinkError e) {
                triedPaths.add("Directly from .apk/lib");
            }
        }

        // Try to load the library from java.library.path
        String javaLibraryPath = System.getProperty("java.library.path", "");
        // String.split's "trailing empties dropped" quirk is benign here because
        // we explicitly skip empty entries with the isEmpty() check below.
        @SuppressWarnings("StringSplitter")
        final String[] ldPaths = javaLibraryPath.split(File.pathSeparator);
        for (String ldPath : ldPaths) {
            if (ldPath.isEmpty()) {
                continue;
            }
            Path path = Paths.get(ldPath, nativeLibName);
            if (loadNativeLibrary(path)) {
                return;
            } else {
                triedPaths.add(ldPath);
            }
        }

        // As a last resort try load the os-dependent library from the jar file
        nativeLibPath = getNativeResourcePath();
        if (hasNativeLib(nativeLibPath, nativeLibName)) {
            // temporary library folder
            String tempFolder = getTempDir().getAbsolutePath();
            // Try extracting the library from jar
            if (extractAndLoadLibraryFile(nativeLibPath, nativeLibName, tempFolder)) {
                return;
            } else {
                triedPaths.add(nativeLibPath);
            }
        }

        throw new UnsatisfiedLinkError(String.format(
                "No native library found for os.name=%s, os.arch=%s, paths=[%s]",
                OSInfo.getOSName(), OSInfo.getArchName(), String.join(File.pathSeparator, triedPaths)));
    }

    /**
     * Loads native library using the given path and name of the library
     *
     * @param path path of the native library
     * @return true for successfully loading, otherwise false
     */
    public static boolean loadNativeLibrary(Path path) {
        if (!Files.exists(path)) {
            return false;
        }
        String absolutePath = path.toAbsolutePath().toString();
        try {
            System.load(absolutePath);
            return true;
        } catch (UnsatisfiedLinkError e) {
            System.err.println(e.getMessage());
            System.err.println("Failed to load native library: " + absolutePath + ". osinfo: "
                    + OSInfo.getNativeLibFolderPathForCurrentOS());
            return false;
        }
    }

    private static @Nullable Path extractFile(String sourceDirectory, String fileName, String targetDirectory) {
        String nativeLibraryFilePath = sourceDirectory + "/" + fileName;

        Path extractedFilePath = Paths.get(targetDirectory, fileName);

        try {
            // Fast path: a byte-identical copy already exists — extracted by a previous run or by a
            // concurrent JVM sharing this tmpdir. Reuse it rather than rewriting in place: replacing a
            // file another process has already loaded fails on Windows (the lib is locked), and an
            // in-place rewrite risks a partial file a concurrent loader could observe.
            if (Files.exists(extractedFilePath) && resourceMatchesFile(nativeLibraryFilePath, extractedFilePath)) {
                permissionSetter.apply(extractedFilePath.toFile());
                extractedFilePath.toFile().deleteOnExit();
                return extractedFilePath;
            }

            // Otherwise extract into a per-attempt unique temp file, verify it, then atomically move it
            // into place so a concurrent loader never observes a half-written library.
            Path tempFile = Files.createTempFile(Paths.get(targetDirectory), fileName + ".", ".tmp");
            try {
                try (InputStream reader = LlamaLoader.class.getResourceAsStream(nativeLibraryFilePath)) {
                    if (reader == null) {
                        return null;
                    }
                    Files.copy(reader, tempFile, StandardCopyOption.REPLACE_EXISTING);
                }
                if (!resourceMatchesFile(nativeLibraryFilePath, tempFile)) {
                    System.err.println(String.format("Failed to write a native library file at %s", extractedFilePath));
                    return null;
                }
                moveIntoPlace(tempFile, extractedFilePath);
            } finally {
                // No-op once moveIntoPlace consumed it; cleans up if any step above bailed out.
                Files.deleteIfExists(tempFile);
            }

            // Set executable (x) flag to enable Java to load the native library.
            permissionSetter.apply(extractedFilePath.toFile());
            extractedFilePath.toFile().deleteOnExit();

            System.out.println("Extracted '" + fileName + "' to '" + extractedFilePath + "'");
            return extractedFilePath;
        } catch (IOException e) {
            System.err.println(e.getMessage());
            return null;
        }
    }

    /**
     * Atomically replace {@code target} with {@code source}, falling back to a plain move when the
     * filesystem does not support atomic moves.
     */
    private static void moveIntoPlace(Path source, Path target) throws IOException {
        try {
            Files.move(source, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (IOException atomicUnsupported) {
            Files.move(source, target, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    /** Whether the classpath resource at {@code resourcePath} is byte-identical to {@code file}. */
    static boolean resourceMatchesFile(String resourcePath, Path file) throws IOException {
        try (InputStream resource = LlamaLoader.class.getResourceAsStream(resourcePath);
                InputStream onDisk = Files.newInputStream(file)) {
            if (resource == null) {
                return false;
            }
            return contentsEquals(resource, onDisk);
        }
    }

    /**
     * Extracts and loads the specified library file to the target folder
     *
     * @param libFolderForCurrentOS Library path.
     * @param libraryFileName       Library name.
     * @param targetFolder          Target folder.
     * @return whether the library was successfully loaded
     */
    private static boolean extractAndLoadLibraryFile(
            String libFolderForCurrentOS, String libraryFileName, String targetFolder) {
        Path path = extractFile(libFolderForCurrentOS, libraryFileName, targetFolder);
        if (path == null) {
            return false;
        }
        return loadNativeLibrary(path);
    }

    static boolean contentsEquals(InputStream in1, InputStream in2) throws IOException {
        if (!(in1 instanceof BufferedInputStream)) {
            in1 = new BufferedInputStream(in1);
        }
        if (!(in2 instanceof BufferedInputStream)) {
            in2 = new BufferedInputStream(in2);
        }

        int ch = in1.read();
        while (ch != -1) {
            int ch2 = in2.read();
            if (ch != ch2) {
                return false;
            }
            ch = in1.read();
        }
        int ch2 = in2.read();
        return ch2 == -1;
    }

    static File getTempDir() {
        String _override = systemProperties.getTmpDir();
        return new File(_override != null ? _override : System.getProperty("java.io.tmpdir"));
    }

    static String getNativeResourcePath() {
        return String.format("%s/%s", NATIVE_RESOURCE_BASE, OSInfo.getNativeLibFolderPathForCurrentOS());
    }

    private static boolean hasNativeLib(String path, String libraryName) {
        return LlamaLoader.class.getResource(path + "/" + libraryName) != null;
    }
}
