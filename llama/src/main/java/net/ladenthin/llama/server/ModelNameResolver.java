// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.jspecify.annotations.Nullable;

/**
 * Resolves a pull spec into a concrete {@link ResolvedModelSource}. Supported specs:
 *
 * <ul>
 *   <li>A direct {@code http(s)://} URL — name and quantization are derived from the file name.</li>
 *   <li>A local file path (absolute, relative, or {@code file:} URI) — registered as-is, no
 *       download.</li>
 *   <li>A curated alias (e.g. {@code llama3.2}, {@code qwen2.5:7b}) looked up in the alias table, via
 *       {@link #ModelNameResolver(Map)} or {@link #loadAliases(Path)}.</li>
 *   <li>A HuggingFace-style {@code org/repo[@quant]} spec, best-effort mapped to
 *       {@code https://huggingface.co/org/repo/resolve/main/repo.gguf}.</li>
 * </ul>
 *
 * <p>Gated HuggingFace models get an {@code Authorization} header from the {@code HF_TOKEN} /
 * {@code HUGGING_FACE_HUB_TOKEN} environment variables when present.
 *
 * <p>Pure Java, no native dependencies. This is the naming/lifecycle layer that Ollama users expect;
 * the resolved URL can later be handed to {@code ModelParameters.setModelUrl} for native-side
 * fetching, but the default {@link ModelPuller} downloads with a pure-Java transport for full control
 * over the on-disk path.
 */
public final class ModelNameResolver {

    private static final Pattern QUANT = Pattern.compile(
            "(BF16|F16|F32|IQ[0-9](?:_[0-9KMSL]+)*|Q[0-9](?:_[0-9KMSL]+)*)", Pattern.CASE_INSENSITIVE);

    private static final String HF_BASE = "https://huggingface.co/";

    private final Map<String, String> aliases;
    private final @Nullable String hfToken;

    /** A resolver with no curated aliases and no HF token. */
    public ModelNameResolver() {
        this(Collections.<String, String>emptyMap(), null);
    }

    /**
     * A resolver with a curated alias table.
     *
     * @param aliases maps friendly names (e.g. {@code llama3.2}) to concrete GGUF URLs
     */
    public ModelNameResolver(Map<String, String> aliases) {
        this(aliases, readToken());
    }

    /**
     * A resolver with explicit aliases and token (mostly for tests).
     *
     * @param aliases curated name → URL map
     * @param hfToken optional HuggingFace token; {@code null} when absent
     */
    public ModelNameResolver(Map<String, String> aliases, @Nullable String hfToken) {
        this.aliases = aliases == null ? Collections.<String, String>emptyMap() : new LinkedHashMap<>(aliases);
        this.hfToken = hfToken;
    }

    /**
     * Load a curated alias table from a JSON file of the shape {@code {"name": "url", ...}}.
     *
     * @param file the alias table file
     * @return a resolver using those aliases and the ambient HF token
     * @throws IOException if the file cannot be read or parsed
     */
    public static ModelNameResolver loadAliases(Path file) throws IOException {
        Map<String, String> map = new LinkedHashMap<>();
        byte[] bytes = Files.readAllBytes(file);
        com.fasterxml.jackson.databind.JsonNode root =
                new com.fasterxml.jackson.databind.ObjectMapper().readTree(new String(bytes, StandardCharsets.UTF_8));
        if (root.isObject()) {
            java.util.Iterator<String> it = root.fieldNames();
            while (it.hasNext()) {
                String key = it.next();
                com.fasterxml.jackson.databind.JsonNode v = root.get(key);
                if (v != null && v.isTextual()) {
                    map.put(key, v.asText());
                }
            }
        }
        return new ModelNameResolver(map);
    }

    private static @Nullable String readToken() {
        String t = System.getenv("HF_TOKEN");
        if (t == null || t.isEmpty()) {
            t = System.getenv("HUGGING_FACE_HUB_TOKEN");
        }
        return (t == null || t.isEmpty()) ? null : t;
    }

    /**
     * Resolve a pull spec.
     *
     * @param spec the URL, local path, alias, or {@code org/repo[@quant]} spec
     * @return the resolved source
     * @throws IllegalArgumentException if the spec is blank or cannot be resolved
     */
    public ResolvedModelSource resolve(String spec) {
        if (spec == null || spec.trim().isEmpty()) {
            throw new IllegalArgumentException("pull spec must not be blank");
        }
        String trimmed = spec.trim();

        if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
            return resolveUrl(trimmed, null);
        }
        if (isLocalPath(trimmed)) {
            return resolveLocal(trimmed);
        }
        return resolveShortName(trimmed);
    }

    private boolean isLocalPath(String spec) {
        if (spec.startsWith("file:") || spec.startsWith("/") || spec.startsWith("./") || spec.startsWith("../")) {
            return true;
        }
        // Windows drive letter, e.g. C:\ or C:/ (avoid misclassifying org/repo as a path).
        if (spec.length() >= 2 && spec.charAt(1) == ':' && Character.isLetter(spec.charAt(0))) {
            return true;
        }
        return Files.exists(java.nio.file.Paths.get(spec));
    }

    private ResolvedModelSource resolveUrl(String url, @Nullable String forcedQuant) {
        String name = nameFromUrl(url);
        String quant = forcedQuant != null ? forcedQuant : quantFromName(name);
        ResolvedModelSource.Builder b =
                new ResolvedModelSource.Builder(name).url(url).quantization(quant);
        if (hfToken != null && url.contains("huggingface.co")) {
            b.headers(Collections.singletonMap("Authorization", "Bearer " + hfToken));
        }
        return b.build();
    }

    private ResolvedModelSource resolveLocal(String spec) {
        Path p = spec.startsWith("file:") ? java.nio.file.Paths.get(java.net.URI.create(spec)) : java.nio.file.Paths.get(spec);
        String name = p.getFileName() == null ? spec : p.getFileName().toString();
        if (name.toLowerCase(java.util.Locale.ROOT).endsWith(".gguf")) {
            name = name.substring(0, name.length() - ".gguf".length());
        }
        return new ResolvedModelSource.Builder(name)
                .localPath(p.toAbsolutePath().toString())
                .quantization(quantFromName(name))
                .build();
    }

    private ResolvedModelSource resolveShortName(String spec) {
        String quant = null;
        String base = spec;
        int at = spec.lastIndexOf('@');
        if (at >= 0) {
            quant = spec.substring(at + 1).trim();
            base = spec.substring(0, at).trim();
        }
        String url = aliases.get(base);
        if (url != null) {
            String name = base;
            int slash = Math.max(url.lastIndexOf('/'), url.lastIndexOf('\\'));
            if (slash >= 0 && slash + 1 < url.length()) {
                name = url.substring(slash + 1);
                if (name.toLowerCase(java.util.Locale.ROOT).endsWith(".gguf")) {
                    name = name.substring(0, name.length() - ".gguf".length());
                }
            }
            return resolveUrl(url, quant != null ? quant : quantFromName(name));
        }
        // Best-effort HuggingFace construction: org/repo -> .../resolve/main/repo.gguf
        if (base.contains("/") && !base.contains(":")) {
            String repoName = base.substring(base.lastIndexOf('/') + 1);
            String hfUrl = HF_BASE + base + "/resolve/main/" + repoName + ".gguf";
            return resolveUrl(hfUrl, quant);
        }
        throw new IllegalArgumentException(
                "Unknown model '" + spec + "'. Use a URL, a local path, or a configured alias "
                        + "(known: " + String.join(", ", aliases.keySet()) + ").");
    }

    private static String nameFromUrl(String url) {
        int q = url.indexOf('?');
        String path = q >= 0 ? url.substring(0, q) : url;
        int slash = Math.max(path.lastIndexOf('/'), path.lastIndexOf('\\'));
        String name = slash >= 0 && slash + 1 < path.length() ? path.substring(slash + 1) : path;
        if (name.toLowerCase(java.util.Locale.ROOT).endsWith(".gguf")) {
            name = name.substring(0, name.length() - ".gguf".length());
        }
        return name;
    }

    private static @Nullable String quantFromName(String name) {
        Matcher m = QUANT.matcher(name);
        if (!m.find()) {
            return null;
        }
        String g = m.group(1);
        return g != null ? g.toUpperCase(java.util.Locale.ROOT) : null;
    }
}
