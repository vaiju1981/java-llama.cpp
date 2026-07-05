// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.langchain4j;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
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
import java.util.Arrays;
import org.junit.jupiter.api.Test;

/** Model-free tests for the langchain4j {@code JsonSchemaElement} &rarr; JSON Schema serializer. */
class JsonSchemaElementSerializerTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private static JsonNode serialize(JsonSchemaElement element) throws IOException {
        return MAPPER.readTree(JsonSchemaElementSerializer.toJson(element));
    }

    @Test
    void serializesObjectWithPrimitivesRequiredAndAdditionalProperties() throws IOException {
        JsonObjectSchema schema = JsonObjectSchema.builder()
                .description("a person")
                .addProperty("name", JsonStringSchema.builder().description("full name").build())
                .addProperty("age", new JsonIntegerSchema())
                .addProperty("height", new JsonNumberSchema())
                .addProperty("active", new JsonBooleanSchema())
                .required("name", "age")
                .additionalProperties(false)
                .build();

        JsonNode node = serialize(schema);

        assertThat(node.path("type").asText(), is("object"));
        assertThat(node.path("description").asText(), is("a person"));
        assertThat(node.path("properties").path("name").path("type").asText(), is("string"));
        assertThat(node.path("properties").path("name").path("description").asText(), is("full name"));
        assertThat(node.path("properties").path("age").path("type").asText(), is("integer"));
        assertThat(node.path("properties").path("height").path("type").asText(), is("number"));
        assertThat(node.path("properties").path("active").path("type").asText(), is("boolean"));
        assertThat(node.path("required").get(0).asText(), is("name"));
        assertThat(node.path("required").get(1).asText(), is("age"));
        assertThat(node.path("additionalProperties").asBoolean(true), is(false));
    }

    @Test
    void serializesEnumAsStringTypeWithValues() throws IOException {
        JsonEnumSchema schema = JsonEnumSchema.builder()
                .description("unit")
                .enumValues("CELSIUS", "FAHRENHEIT")
                .build();

        JsonNode node = serialize(schema);

        // langchain4j convention: enums are string-typed with an "enum" list.
        assertThat(node.path("type").asText(), is("string"));
        assertThat(node.path("enum").get(0).asText(), is("CELSIUS"));
        assertThat(node.path("enum").get(1).asText(), is("FAHRENHEIT"));
    }

    @Test
    void serializesArrayWithItems() throws IOException {
        JsonArraySchema schema = JsonArraySchema.builder()
                .description("tags")
                .items(new JsonStringSchema())
                .build();

        JsonNode node = serialize(schema);

        assertThat(node.path("type").asText(), is("array"));
        assertThat(node.path("items").path("type").asText(), is("string"));
    }

    @Test
    void serializesReferenceAndDefinitionsInLangchainConvention() throws IOException {
        // Recursive shape: a node whose children reference the node definition itself.
        JsonObjectSchema schema = JsonObjectSchema.builder()
                .addProperty(
                        "root",
                        JsonReferenceSchema.builder().reference("TreeNode").build())
                .definitions(java.util.Collections.singletonMap(
                        "TreeNode",
                        JsonObjectSchema.builder()
                                .addProperty("value", new JsonStringSchema())
                                .build()))
                .build();

        JsonNode node = serialize(schema);

        // Must match langchain4j's internal convention so schema round-trips are stable.
        assertThat(node.path("properties").path("root").path("$ref").asText(), is("#/$defs/TreeNode"));
        assertThat(
                node.path("$defs").path("TreeNode").path("properties").path("value").path("type").asText(),
                is("string"));
    }

    @Test
    void serializesAnyOfIncludingNull() throws IOException {
        JsonAnyOfSchema schema = JsonAnyOfSchema.builder()
                .anyOf(Arrays.asList(new JsonStringSchema(), new JsonNullSchema()))
                .build();

        JsonNode node = serialize(schema);

        assertThat(node.path("anyOf").get(0).path("type").asText(), is("string"));
        assertThat(node.path("anyOf").get(1).path("type").asText(), is("null"));
    }

    @Test
    void passesRawSchemaThroughVerbatim() throws IOException {
        JsonRawSchema schema = JsonRawSchema.from("{\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"integer\"}}}");

        JsonNode node = serialize(schema);

        assertThat(node.path("properties").path("x").path("type").asText(), is("integer"));
    }

    @Test
    void rejectsUnparseableRawSchema() {
        JsonRawSchema broken = JsonRawSchema.from("not json");

        assertThrows(IllegalArgumentException.class, () -> JsonSchemaElementSerializer.toJson(broken));
    }

    @Test
    void objectWithoutPropertiesStillCarriesEmptyPropertiesObject() throws IOException {
        JsonNode node = serialize(JsonObjectSchema.builder().build());

        // The OAI tools array expects "parameters" to be a full (if empty) object schema.
        assertThat(node.path("type").asText(), is("object"));
        assertThat(node.path("properties").isObject(), is(true));
        assertThat(node.path("properties").size(), is(0));
    }
}
