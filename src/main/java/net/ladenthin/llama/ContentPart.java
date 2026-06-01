// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;
import java.util.Locale;
import java.util.Objects;
import org.jspecify.annotations.Nullable;

/**
 * One piece of a {@link ChatMessage}'s multimodal content array: either a text
 * fragment or an image URL (typically a {@code data:image/...;base64,...} URI).
 * Mirrors the OpenAI-compatible {@code content} part shape that the upstream
 * {@code llama.cpp} server already understands, so no new JNI plumbing is
 * required &#x2014; an image-bearing message is serialized to
 * <pre>
 * {"role":"user","content":[
 *   {"type":"text","text":"What is in this image?"},
 *   {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}
 * ]}
 * </pre>
 * and the upstream {@code oaicompat_chat_params_parse} routes it through the
 * compiled-in {@code mtmd} pipeline (requires
 * {@link ModelParameters#setMmproj(String)} to be wired).
 * <p>
 * Instances are immutable and safe to share across threads. Use the static
 * factories &#x2014; the constructor is private.
 * </p>
 */
public final class ContentPart {

    /** Discriminator for the two part kinds the OAI multipart schema supports. */
    public enum Type {
        /** A plain-text fragment. */
        TEXT,
        /** An image reference (data URI or remote URL). */
        IMAGE_URL
    }

    private final Type type;
    private final @Nullable String text;
    private final @Nullable String imageUrl;

    private ContentPart(Type type, @Nullable String text, @Nullable String imageUrl) {
        this.type = type;
        this.text = text;
        this.imageUrl = imageUrl;
    }

    /**
     * Build a text part.
     *
     * @param text the text fragment (must not be {@code null})
     * @return a TEXT part wrapping {@code text}
     */
    public static ContentPart text(String text) {
        Objects.requireNonNull(text, "text");
        return new ContentPart(Type.TEXT, text, null);
    }

    /**
     * Build an image part from a pre-formed URL or data URI. Pass either an
     * HTTP(S) URL (if the server is configured to fetch it) or a complete
     * {@code data:image/...;base64,...} string.
     *
     * @param url image URL or data URI (must not be {@code null})
     * @return an IMAGE_URL part wrapping {@code url}
     */
    public static ContentPart imageUrl(String url) {
        Objects.requireNonNull(url, "url");
        return new ContentPart(Type.IMAGE_URL, null, url);
    }

    /**
     * Build an image part from raw bytes plus an explicit MIME type. The bytes
     * are base64-encoded and wrapped in a {@code data:} URI.
     *
     * @param bytes raw image bytes (must not be {@code null})
     * @param mimeType MIME type, e.g. {@code "image/png"} (must not be {@code null} or empty)
     * @return an IMAGE_URL part carrying the data URI
     */
    public static ContentPart imageBytes(byte[] bytes, String mimeType) {
        Objects.requireNonNull(bytes, "bytes");
        Objects.requireNonNull(mimeType, "mimeType");
        if (mimeType.isEmpty()) {
            throw new IllegalArgumentException("mimeType must not be empty");
        }
        String encoded = Base64.getEncoder().encodeToString(bytes);
        return new ContentPart(Type.IMAGE_URL, null, "data:" + mimeType + ";base64," + encoded);
    }

    /**
     * Build an image part by reading a file from disk and detecting its MIME
     * type from the file extension. Recognised extensions: {@code .png},
     * {@code .jpg}, {@code .jpeg}, {@code .webp}, {@code .gif}. Anything else
     * throws {@link IllegalArgumentException}; use {@link #imageBytes(byte[], String)}
     * to force a MIME type explicitly.
     *
     * @param imagePath path to the image file (must not be {@code null})
     * @return an IMAGE_URL part carrying the data URI
     * @throws IOException if the file cannot be read
     */
    public static ContentPart imageFile(Path imagePath) throws IOException {
        Objects.requireNonNull(imagePath, "imagePath");
        Path fileNamePath = imagePath.getFileName();
        if (fileNamePath == null) {
            throw new IllegalArgumentException("imagePath has no file name component: " + imagePath);
        }
        String name = fileNamePath.toString().toLowerCase(Locale.ROOT);
        String mimeType;
        if (name.endsWith(".png")) {
            mimeType = "image/png";
        } else if (name.endsWith(".jpg") || name.endsWith(".jpeg")) {
            mimeType = "image/jpeg";
        } else if (name.endsWith(".webp")) {
            mimeType = "image/webp";
        } else if (name.endsWith(".gif")) {
            mimeType = "image/gif";
        } else {
            throw new IllegalArgumentException("Cannot infer MIME type from extension: " + imagePath
                    + " &#x2014; use ContentPart.imageBytes(bytes, mimeType) instead");
        }
        return imageBytes(Files.readAllBytes(imagePath), mimeType);
    }

    /**
     * Part-kind accessor.
     * @return the discriminator selecting {@link #getText()} or {@link #getImageUrl()}
     */
    public Type getType() {
        return type;
    }

    /**
     * Text accessor (only set for {@link Type#TEXT}).
     * @return the text fragment, or {@code null} for {@link Type#IMAGE_URL} parts
     */
    public @Nullable String getText() {
        return text;
    }

    /**
     * Image URL accessor (only set for {@link Type#IMAGE_URL}).
     * @return the URL or data URI, or {@code null} for {@link Type#TEXT} parts
     */
    public @Nullable String getImageUrl() {
        return imageUrl;
    }
}
