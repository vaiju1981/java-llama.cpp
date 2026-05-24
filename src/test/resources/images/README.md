# Test resource images

## `test-image.jpg`

A 960×720 JPEG (~104 KB) used by `MultimodalIntegrationTest` to exercise
the typed multimodal request path end-to-end without a network
dependency.

### Provenance

This is a copy of the Wikimedia Commons photo
[`File:20200601_135745_Flowers_and_Bees.jpg`](https://commons.wikimedia.org/wiki/File:20200601_135745_Flowers_and_Bees.jpg)
by Bernard Ladenthin (project copyright holder). Captured on an Olympus
digital camera; the JPEG's EXIF copyright tag confirms authorship.

### License

The Commons page lists the photo under CC-BY-4.0. The same author
additionally grants this file for use in this project under the
project's MIT license, so no runtime attribution machinery is needed.
A `SPDX-FileCopyrightText: 2026 Bernard Ladenthin
<bernard.ladenthin@gmail.com>` line in the commit that added this file
records that grant.

### Override

To point the test at a different image without overwriting this one,
set the `net.ladenthin.llama.vision.image` system property on the
`mvn test` command line. Any image the JRE can read works, provided the
file extension matches one of `.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`
so `ContentPart.imageFile(Path)` resolves the MIME type.
