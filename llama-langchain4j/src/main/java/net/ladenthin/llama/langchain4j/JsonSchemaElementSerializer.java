// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.langchain4j;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import dev.langchain4j.model.chat.request.json.JsonAnyOfSchema;
import dev.langchain4j.model.chat.request.json.JsonArraySchema;
import dev.langchain4j.model.chat.request.json.JsonBooleanSchema;
import dev.langchain4j.model.chat.request.json.JsonEnumSchema;
import dev.langchain4j.model.chat.request.json.JsonIntegerSchema;
import dev.langchain4j.model.chat.request.json.JsonNullSchema;
import dev.langchain4j.model.chat.request.json.JsonNumberSchema;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.chat.request.json.JsonRawSchema;
import dev.langchain4j.model.chat.request.json.JsonReferenceSchema;
import dev.langchain4j.model.chat.request.json.JsonSchemaElement;
import dev.langchain4j.model.chat.request.json.JsonStringSchema;
import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * Serializes a langchain4j {@link JsonSchemaElement} tree to a standard JSON Schema string.
 *
 * <p>langchain4j's own serializer lives in its {@code internal} package (not public API), so this
 * module carries its own recursive walk over the public element types. The emitted shape follows
 * the langchain4j conventions so schemas produced by langchain4j's annotation processing round-trip
 * unchanged: object definitions land under {@code $defs}, and a {@link JsonReferenceSchema} becomes
 * {@code {"$ref": "#/$defs/&lt;reference&gt;"}}.
 *
 * <p>Pure data transform: no JNI, no model state, unit-testable with schema literals (see
 * {@code JsonSchemaElementSerializerTest}).
 */
final class JsonSchemaElementSerializer {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private JsonSchemaElementSerializer() {}

    /**
     * Serialize a schema element tree to its JSON Schema string form.
     *
     * @param element the root element (must not be {@code null})
     * @return the JSON Schema as a string
     * @throws IllegalArgumentException on an unknown element type or an unparseable
     *     {@link JsonRawSchema}
     */
    static String toJson(JsonSchemaElement element) {
        return toJsonNode(element).toString();
    }

    /**
     * Serialize a schema element tree to a Jackson node.
     *
     * @param element the root element (must not be {@code null})
     * @return the JSON Schema as an {@link ObjectNode}
     * @throws IllegalArgumentException on an unknown element type or an unparseable
     *     {@link JsonRawSchema}
     */
    static ObjectNode toJsonNode(JsonSchemaElement element) {
        if (element instanceof JsonObjectSchema) {
            return objectNode((JsonObjectSchema) element);
        }
        if (element instanceof JsonStringSchema) {
            return primitiveNode("string", ((JsonStringSchema) element).description());
        }
        if (element instanceof JsonIntegerSchema) {
            return primitiveNode("integer", ((JsonIntegerSchema) element).description());
        }
        if (element instanceof JsonNumberSchema) {
            return primitiveNode("number", ((JsonNumberSchema) element).description());
        }
        if (element instanceof JsonBooleanSchema) {
            return primitiveNode("boolean", ((JsonBooleanSchema) element).description());
        }
        if (element instanceof JsonEnumSchema) {
            return enumNode((JsonEnumSchema) element);
        }
        if (element instanceof JsonArraySchema) {
            return arrayNode((JsonArraySchema) element);
        }
        if (element instanceof JsonReferenceSchema) {
            return referenceNode((JsonReferenceSchema) element);
        }
        if (element instanceof JsonAnyOfSchema) {
            return anyOfNode((JsonAnyOfSchema) element);
        }
        if (element instanceof JsonNullSchema) {
            return primitiveNode("null", ((JsonNullSchema) element).description());
        }
        if (element instanceof JsonRawSchema) {
            return rawNode((JsonRawSchema) element);
        }
        throw new IllegalArgumentException(
                "Unsupported JsonSchemaElement type: " + element.getClass().getName());
    }

    private static ObjectNode objectNode(JsonObjectSchema schema) {
        ObjectNode node = MAPPER.createObjectNode();
        node.put("type", "object");
        putDescription(node, schema.description());
        ObjectNode properties = node.putObject("properties");
        Map<String, JsonSchemaElement> schemaProperties = schema.properties();
        if (schemaProperties != null) {
            for (Map.Entry<String, JsonSchemaElement> entry : schemaProperties.entrySet()) {
                properties.set(entry.getKey(), toJsonNode(entry.getValue()));
            }
        }
        List<String> required = schema.required();
        if (required != null && !required.isEmpty()) {
            ArrayNode requiredNode = node.putArray("required");
            for (String name : required) {
                requiredNode.add(name);
            }
        }
        if (schema.additionalProperties() != null) {
            node.put("additionalProperties", schema.additionalProperties().booleanValue());
        }
        Map<String, JsonSchemaElement> definitions = schema.definitions();
        if (definitions != null && !definitions.isEmpty()) {
            ObjectNode defs = node.putObject("$defs");
            for (Map.Entry<String, JsonSchemaElement> entry : definitions.entrySet()) {
                defs.set(entry.getKey(), toJsonNode(entry.getValue()));
            }
        }
        return node;
    }

    private static ObjectNode enumNode(JsonEnumSchema schema) {
        // langchain4j emits enums as string-typed with an "enum" values list.
        ObjectNode node = primitiveNode("string", schema.description());
        ArrayNode values = node.putArray("enum");
        List<String> enumValues = schema.enumValues();
        if (enumValues != null) {
            for (String value : enumValues) {
                values.add(value);
            }
        }
        return node;
    }

    private static ObjectNode arrayNode(JsonArraySchema schema) {
        ObjectNode node = MAPPER.createObjectNode();
        node.put("type", "array");
        putDescription(node, schema.description());
        if (schema.items() != null) {
            node.set("items", toJsonNode(schema.items()));
        }
        return node;
    }

    private static ObjectNode referenceNode(JsonReferenceSchema schema) {
        ObjectNode node = MAPPER.createObjectNode();
        // Mirrors langchain4j's internal convention: definitions live under "$defs".
        if (schema.reference() != null) {
            node.put("$ref", "#/$defs/" + schema.reference());
        }
        return node;
    }

    private static ObjectNode anyOfNode(JsonAnyOfSchema schema) {
        ObjectNode node = MAPPER.createObjectNode();
        putDescription(node, schema.description());
        ArrayNode anyOf = node.putArray("anyOf");
        List<JsonSchemaElement> elements = schema.anyOf();
        if (elements != null) {
            for (JsonSchemaElement element : elements) {
                anyOf.add(toJsonNode(element));
            }
        }
        return node;
    }

    private static ObjectNode rawNode(JsonRawSchema schema) {
        try {
            return (ObjectNode) MAPPER.readTree(schema.schema());
        } catch (IOException | ClassCastException e) {
            throw new IllegalArgumentException("JsonRawSchema does not contain a JSON object: " + schema.schema(), e);
        }
    }

    private static ObjectNode primitiveNode(String type, String description) {
        ObjectNode node = MAPPER.createObjectNode();
        node.put("type", type);
        putDescription(node, description);
        return node;
    }

    private static void putDescription(ObjectNode node, String description) {
        if (description != null) {
            node.put("description", description);
        }
    }
}
