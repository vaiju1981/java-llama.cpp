// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import java.util.HashMap;
import java.util.Map;
import net.ladenthin.llama.args.CliArg;
import net.ladenthin.llama.json.ParameterJsonSerializer;
import org.checkerframework.checker.nullness.qual.PolyNull;

/**
 * The Java library re-uses most of the llama.cpp server code, which mostly works with JSONs. Thus, the complexity and
 * maintainability is much lower if we work with JSONs. This class provides a simple abstraction to easily create
 * JSON object strings by filling a <code>Map&lt;String, String&gt;</code> with key value pairs.
 */
abstract class JsonParameters {

    // We save parameters directly as a String map here, to re-use as much as possible of the (json-based) C++ code.
    // The JNI code for a proper Java-typed data object is comparatively too complex and hard to maintain.
    final Map<String, String> parameters = new HashMap<>();

    /** Serializer for converting Java values to JSON-safe strings. */
    protected final ParameterJsonSerializer serializer = new ParameterJsonSerializer();

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("{\n");
        int i = 0;
        for (Map.Entry<String, String> entry : parameters.entrySet()) {
            String key = entry.getKey();
            String value = entry.getValue();
            builder.append("\t\"").append(key).append("\": ").append(value);
            if (i++ < parameters.size() - 1) {
                builder.append(',');
            }
            builder.append('\n');
        }
        builder.append('}');
        return builder.toString();
    }

    // @PolyNull lets the Checker Framework see that null in returns null and non-null
    // in returns non-null. NullAway has no equivalent qualifier and reads the return as
    // @NonNull (under @NullMarked), so we suppress the NullAway-only complaint here.
    @SuppressWarnings("NullAway")
    @PolyNull String toJsonString(@PolyNull String text) {
        if (text == null) return null;
        return serializer.toJsonString(text);
    }

    /**
     * Store a scalar value (typically a primitive: int, long, float, double, boolean)
     * for the given key using {@link String#valueOf(Object)} and return this builder
     * typed as the concrete subtype so callers can collapse the
     * {@code parameters.put(...); return this;} pair into a single
     * {@code return putScalar(...);}.
     *
     * @param key   the parameter key
     * @param value the scalar value; autoboxed at the call site
     * @param <T>   the concrete subtype of this builder
     * @return this builder
     */
    // Self-typing builder idiom: the caller fixes T to its own concrete subtype
    // so that chained calls return the concrete builder instead of JsonParameters.
    // This deliberately uses T only in the return type and is not the
    // "TypeParameterUnusedInFormals" anti-pattern Error Prone warns about.
    @SuppressWarnings({"unchecked", "TypeParameterUnusedInFormals"})
    protected final <T extends JsonParameters> T putScalar(String key, Object value) {
        parameters.put(key, String.valueOf(value));
        return (T) this;
    }

    /**
     * Store the CLI-argument string of the given enum constant for the given key and
     * return this builder typed as the concrete subtype.
     *
     * @param key   the parameter key
     * @param value the enum constant; must implement {@link CliArg}
     * @param <T>   the concrete subtype of this builder
     * @return this builder
     */
    // Self-typing builder idiom — see putScalar above.
    @SuppressWarnings({"unchecked", "TypeParameterUnusedInFormals"})
    protected final <T extends JsonParameters> T putEnum(String key, CliArg value) {
        parameters.put(key, value.getArgValue());
        return (T) this;
    }
}
