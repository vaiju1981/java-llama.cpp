<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT
-->
# LLM Service — store assets

Assets used when **publishing** the app, **not** bundled into the APK/AAB. Keep them
here so a Play Console upload has everything in one place.

| File | Use |
|---|---|
| `ic_launcher-512.png` | Google Play **hi-res app icon** (512×512 PNG, 32-bit). Uploaded in the Play Console store listing — it is required there and is separate from the on-device launcher icon under `app/src/main/res/mipmap-*/`. |

The icon is the **shard-`J`** of the java-llama.cpp logo, from the shared logo set in
the [`workspace`](https://github.com/bernardladenthin/workspace) repo
(`llama.cpp/projects/logo/app-icon/`). Licensed **MIT OR Apache-2.0**.
