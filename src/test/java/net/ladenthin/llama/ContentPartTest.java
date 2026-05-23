// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;

import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.Rule;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

@ClaudeGenerated(
        purpose = "Factory contracts and data-URI shape for the multimodal ContentPart value type."
)
public class ContentPartTest {

    @Rule
    public TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void textPartCarriesText() {
        ContentPart p = ContentPart.text("hello");
        assertEquals(ContentPart.Type.TEXT, p.getType());
        assertEquals("hello", p.getText());
        assertNull(p.getImageUrl());
    }

    @Test
    public void imageUrlPartCarriesUrl() {
        ContentPart p = ContentPart.imageUrl("https://example.com/a.png");
        assertEquals(ContentPart.Type.IMAGE_URL, p.getType());
        assertEquals("https://example.com/a.png", p.getImageUrl());
        assertNull(p.getText());
    }

    @Test
    public void imageBytesProducesDataUri() {
        byte[] bytes = {1, 2, 3, 4, 5};
        ContentPart p = ContentPart.imageBytes(bytes, "image/png");
        String expected = "data:image/png;base64," + Base64.getEncoder().encodeToString(bytes);
        assertEquals(expected, p.getImageUrl());
    }

    @Test(expected = NullPointerException.class)
    public void textRejectsNull() {
        ContentPart.text(null);
    }

    @Test(expected = NullPointerException.class)
    public void imageUrlRejectsNull() {
        ContentPart.imageUrl(null);
    }

    @Test(expected = NullPointerException.class)
    public void imageBytesRejectsNullBytes() {
        ContentPart.imageBytes(null, "image/png");
    }

    @Test(expected = NullPointerException.class)
    public void imageBytesRejectsNullMimeType() {
        ContentPart.imageBytes(new byte[]{0}, null);
    }

    @Test(expected = IllegalArgumentException.class)
    public void imageBytesRejectsEmptyMimeType() {
        ContentPart.imageBytes(new byte[]{0}, "");
    }

    @Test
    public void imageFileDetectsPngMime() throws IOException {
        Path file = tmp.newFile("logo.PNG").toPath();
        Files.write(file, new byte[]{(byte) 0x89, 0x50, 0x4E, 0x47});
        ContentPart p = ContentPart.imageFile(file);
        assertTrue(p.getImageUrl().startsWith("data:image/png;base64,"));
    }

    @Test
    public void imageFileDetectsJpegFromJpgExtension() throws IOException {
        Path file = tmp.newFile("photo.jpg").toPath();
        Files.write(file, new byte[]{(byte) 0xFF, (byte) 0xD8, (byte) 0xFF});
        ContentPart p = ContentPart.imageFile(file);
        assertTrue(p.getImageUrl().startsWith("data:image/jpeg;base64,"));
    }

    @Test
    public void imageFileDetectsJpegFromJpegExtension() throws IOException {
        Path file = tmp.newFile("photo.jpeg").toPath();
        Files.write(file, new byte[]{(byte) 0xFF, (byte) 0xD8, (byte) 0xFF});
        ContentPart p = ContentPart.imageFile(file);
        assertTrue(p.getImageUrl().startsWith("data:image/jpeg;base64,"));
    }

    @Test
    public void imageFileDetectsWebp() throws IOException {
        Path file = tmp.newFile("img.webp").toPath();
        Files.write(file, new byte[]{0x52, 0x49, 0x46, 0x46});
        ContentPart p = ContentPart.imageFile(file);
        assertTrue(p.getImageUrl().startsWith("data:image/webp;base64,"));
    }

    @Test
    public void imageFileDetectsGif() throws IOException {
        Path file = tmp.newFile("anim.gif").toPath();
        Files.write(file, new byte[]{0x47, 0x49, 0x46, 0x38});
        ContentPart p = ContentPart.imageFile(file);
        assertTrue(p.getImageUrl().startsWith("data:image/gif;base64,"));
    }

    @Test
    public void imageFileRejectsUnknownExtension() throws IOException {
        Path file = tmp.newFile("doc.txt").toPath();
        Files.write(file, "hello".getBytes());
        try {
            ContentPart.imageFile(file);
            fail("expected IllegalArgumentException for unknown extension");
        } catch (IllegalArgumentException expected) {
            assertNotNull(expected.getMessage());
        }
    }
}
