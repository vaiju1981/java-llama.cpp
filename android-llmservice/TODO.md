<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT
-->
# LLM Service — feature roadmap / TODO

Feature backlog for the **LLM Service** Android app (`android-llmservice/`), derived from the
"Local AI / Offline LLM App" design brief and mapped against what the `net.ladenthin:llama-android`
AAR + `net.ladenthin:llama-kotlin` façade actually make possible today.

**Effort (t-shirt):** `XS` < ½ day · `S` ~1 day · `M` 2–4 days · `L` ~1–2 weeks · `XL` multi-week /
needs foundational native or upstream work.
**Status:** ✅ done · 🔧 partial · ⬜ todo · 🚧 blocked (needs foundational work) · 🚫 out of scope.

## Read this first — two decisions that gate large chunks of the brief

1. **The "Server" screen is the biggest item and is currently _blocked_ on Android.** The design
   brief's flagship differentiator (a local OpenAI/Ollama-compatible API server + WebUI in the
   browser) does **not** work out-of-the-box on Android today: `NativeServer` (the full upstream
   `llama_server` + WebUI) is compiled out on Android (it pulls in `subprocess.h`/`posix_spawn_*`),
   and `OpenAiCompatServer` needs `com.sun.net.httpserver`, which is not part of the Android runtime.
   The cpp-httplib HTTP backend _is_ already compiled into the Android `libjllama.so`, but no
   startable entry point is exposed there. **Recommended path:** add a pure-Kotlin **Ktor** (or
   NanoHTTPD) HTTP server _inside the app_ that translates OpenAI/Ollama requests onto the existing
   JNI inference core (`LlamaModel.streamChatCompletion` / `handleChatCompletions` / `handle*`). That
   sidesteps both blockers entirely and reuses everything the façade already exposes. Estimated `XL`.

2. **Model download needs the `INTERNET` permission** — a deliberate departure from the app's current
   **zero-network** posture (no `INTERNET` permission at all). The brief acknowledges this
   ("Internet is only needed for model downloads"). Keep offline-by-default and scope network access
   strictly to the downloader, but this is a conscious privacy-posture change and needs a yes/no.

Everything else in the brief is feasible with standard AndroidX + the existing binding API.

---

## Already shipped (baseline)

| Feature | Status | Note |
|---|---|---|
| On-device streaming chat (Compose) | ✅ | `generateChatFlow` façade; CPU inference |
| Pick GGUF from file system (SAF) | ✅ | copied into private `filesDir` (llama.cpp mmaps a real path) |
| 13-language i18n + in-app flag language picker | ✅ | AppCompat per-app locales, persisted |
| Private local save / load session | ✅ | JSON in `filesDir`, app-private, nothing uploaded |
| `allowBackup=false` (data stays on device) | ✅ | resolves CodeQL alert; privacy posture |
| minSdk 28, arm64-v8a + x86_64, signed release `.aab`, on-device emulator UI test | ✅ | |
| Generation settings sheet (temperature, Top-K/P, Min-P, **repeat penalty**, repeat range, max tokens) | ✅ | ⚙️ in the top bar; live sliders + reset. The repeat-penalty default (1.1) fixes the degenerate repetition loop on small models |
| In-app log: always-visible one-line strip + full viewer (copy-all, save-as-txt via SAF, clear) | ✅ | 🧾 strip at the bottom; tap to open, ✕ to close |
| Draggable (horizontally scrollable) model-name title | ✅ | long names can be dragged into view — no auto-marquee wobble |

---

## 1. App shell, navigation & theme

| Feature | Purpose | Effort | Status | Notes |
|---|---|---|---|---|
| Bottom navigation (Chat / Models / Server / Settings) | Core IA from the brief | M | ⬜ | restructures today's single screen into 4 destinations (Navigation-Compose) |
| Material 3 theme + brand palette (indigo `#476EAC`, lavender, charcoal, cream) | Polished, on-brand look | S–M | 🔧 | currently default `MaterialTheme`; add color scheme + shapes/elevation |
| Light + dark ("technical dark" / "soft light") | Brief's hybrid direction | S | 🔧 | Compose supports; wire dynamic + manual toggle |
| Smooth transitions / motion polish | "more polished than a dev demo" | S | ⬜ | Compose animations, shared-element nav |
| Optional 5th destination: Tools / Logs | Advanced users | S | 🔧 | **log** shipped as a bottom strip + full viewer (copy/save-txt/clear); a dedicated Tools destination is still todo |

## 2. Onboarding & device readiness

| Feature | Purpose | Effort | Status | Notes |
|---|---|---|---|---|
| 3-card onboarding (Private by Design / Choose model / Chat or Serve) | First-run trust + orientation | S–M | ⬜ | one-time, persisted flag |
| Device readiness check | Set expectations before a model fails | M | ⬜ | see rows below; single screen after onboarding |
| Free RAM | Recommend model size | S | ⬜ | `ActivityManager.MemoryInfo` (`availMem`/`totalMem`) |
| Free storage | Can the model be stored? | XS | ⬜ | `StatFs` on `filesDir` |
| Battery level / charging | Warn on heavy inference | XS | ⬜ | `BatteryManager` |
| GPU acceleration available? | Show if Adreno/OpenCL usable | M | ⬜ | only the OpenCL/Adreno AAR path (see §7); detect ICD presence |
| Recommended model size heuristic | Approachable guidance | S | ⬜ | derive from free RAM + quant table |
| Offline-mode status badge | Reinforce privacy | XS | ✅/🔧 | app is already offline; surface it |

## 3. Chat screen

| Feature | Purpose | Effort | Status | Notes |
|---|---|---|---|---|
| Streaming responses + bubbles | Core chat | S | ✅ | polish to pastel user bubbles / larger AI cards |
| Top bar: model chip · "Local" · RAM indicator · offline badge · switch | Context + control | M | 🔧 | model name shown (now horizontally draggable so long names are fully readable); RAM/switch/offline todo |
| Markdown rendering (code, lists, tables, headings) | Readable answers | M | ⬜ | add a Compose Markdown renderer (e.g. compose-markdown) |
| Copy / regenerate / stop / clear-chat actions | Standard chat UX | M | 🔧 | **stop** = `CancellationToken` (façade wires it; `completeSuspend`/flow cancel) |
| Prompt shortcut chips (Summarize / Explain code / Translate / …) | Fast starts | S | ⬜ | localized prompt templates |
| Attachment button → image input (vision) | Multimodal chat | L | ⬜ | `ContentPart.imageFile(...)` + a **vision GGUF + mmproj**; SAF image picker |
| Audio input (speech) | Multimodal chat | L | ⬜ | `ContentPart.audioFile(...)` + an audio-capable model + mmproj |
| Explicit states: loading / generating / stopped / unavailable / "too large for RAM" | Transparency | S | 🔧 | basic states exist; add the rest + messages |

## 4. Model manager (the differentiator)

| Feature | Purpose | Effort | Status | Notes |
|---|---|---|---|---|
| Model cards (name, params, quant, size, RAM est., modality, status, compat badge, actions) | Curated local library | L | ⬜ | metadata via **`GgufInspector`** for imported files; catalog for downloadable |
| Curated model catalog (bundled) | "not a raw file list" | M | ⬜ | ship a JSON of recommended models (name/size/quant/RAM/modality/url) |
| Import GGUF into a managed library | Bring-your-own model | M | 🔧 | already have SAF picker; extend to a persistent library dir + list |
| Multiple stored models + switch active | Real model manager | M | ⬜ | persist library, select/load one at a time (RAM-bound) |
| RAM estimate + compatibility badges ("Recommended / May be slow / Too large") | Approachable guidance | S–M | ⬜ | heuristic from file size + quant + free RAM |
| Filters (Recommended / Small / Vision / GGUF / Downloaded) | Findability | S | ⬜ | client-side over the catalog + library |
| Delete model / delete unused | Storage hygiene | S | ⬜ | red / destructive confirm |
| **Download manager (HuggingFace)** | Get models in-app | L | ⬜ 🚧 | **needs `INTERNET` permission** (posture change, decision #2); resumable download + progress + checksum |
| TFLite / LiteRT models | (brief mentions it) | — | 🚫 | **out of scope** — this app is GGUF/llama.cpp; LiteRT is a different runtime |

## 5. Server screen (OpenAI/Ollama-compatible) — **needs foundational work**

See decision #1. All rows below assume a new in-app Ktor/NanoHTTPD HTTP layer wrapping the JNI core.

| Feature | Purpose | Effort | Status | Notes |
|---|---|---|---|---|
| Local API server core (start/stop, endpoint, loaded model) | The flagship differentiator | XL | 🚧 | Ktor server → `LlamaModel.streamChatCompletion`/`handle*`; localhost bind |
| OpenAI-compatible routes (`/v1/chat/completions`, …) | Editor/tool integration | (part of XL) | 🚧 | reuse the façade's OAI mappers where possible |
| Ollama-compatible routes (`/api/chat`, `/api/tags`, …) | Ollama clients | M | 🚧 | on top of the core |
| API key (on by default) | Security default | S | 🚧 | |
| LAN access toggle (off by default) + warning | Expose to Wi-Fi safely | S | 🚧 | bind `0.0.0.0` only when enabled; explain reachability |
| WebUI in browser | "open in Chrome" story | M | 🚧 | serve the llama.cpp WebUI (assets already compiled into `libjllama`) or a lightweight own UI |
| Copy endpoint / copy API key / test request / view logs / reset port | Dev ergonomics | S–M | 🚧 | |
| Foreground service + notification (keep server alive) | Reliability | M | 🚧 | Android foreground-service requirement for a long-running server |

## 6. Settings

**Privacy**

| Feature | Purpose | Effort | Status | Notes |
|---|---|---|---|---|
| Offline mode toggle | Reinforce/lock offline | XS | 🔧 | app is already networkless; make it explicit |
| Clear chat history | Control | XS | ⬜ | complements save/load |
| Encrypt local conversations | Protect the saved session | M | ⬜ | Jetpack `security-crypto` `EncryptedFile` |
| "No analytics / no tracking" transparency | Trust | XS | ✅ | trivially true today; surface it |
| Require biometric unlock | Protect the app | M | ⬜ | `androidx.biometric` |

**Performance** (all map to existing `ModelParameters`/`InferenceParameters`)

| Feature | Purpose | Effort | Status | Notes |
|---|---|---|---|---|
| CPU threads | Tune speed | S | ⬜ | `ModelParameters.setThreads` / `setThreadsBatch` |
| Context length | Memory vs. history | S | ⬜ | `setCtxSize` |
| Temperature / Max tokens / Top-K / Top-P / Min-P / **repeat penalty** + range | Sampling control | S | ✅ | shipped in the ⚙️ Generation settings sheet (live sliders + reset); `withTemperature`/`withNPredict`/`withTopK`/`withTopP`/`withMinP`/`withRepeatPenalty`/`withRepeatLastN`. The repeat-penalty default is what stops the small-model repetition loop |
| GPU acceleration toggle | Speed on Adreno | M | 🔧 | `setGpuLayers`; **Adreno/OpenCL AAR only** (see §7) |
| Battery saver mode | Protect battery | M | ⬜ | throttle threads / ngl on low battery |
| Thermal protection | Prevent overheating | M | ⬜ | `PowerManager.getCurrentThermalStatus()` → pause/throttle |

**Storage**

| Feature | Purpose | Effort | Status | Notes |
|---|---|---|---|---|
| Model folder view | Where models live | S | ⬜ | managed library dir |
| Cache size / clear cache | Reclaim space | S | ⬜ | |
| Import GGUF | Bring-your-own | XS | 🔧 | picker exists |

**Developer** — the OpenAI/Ollama API, port, API key, WebUI, logs rows all live under §5 (blocked).

## 7. Platform / acceleration notes

| Item | Reality today | Effort to adopt |
|---|---|---|
| GPU acceleration | **OpenCL/Adreno only** via the separate `net.ladenthin:llama-android-opencl` AAR (Qualcomm Snapdragon/Adreno; device must supply an OpenCL ICD). **No Vulkan** Android artifact exists. | M — swap/offer the OpenCL AAR flavor + runtime detection; non-Adreno devices stay CPU |
| Vision / audio input | Supported by the binding (`ContentPart.imageFile`/`audioFile`), needs a multimodal GGUF + matching mmproj | L (see §3) |
| 16 KB page size / dlopen-clean `.so` | Already guaranteed by the AAR (CI-enforced) | — |

## 8. Branding & store

| Feature | Purpose | Effort | Status | Notes |
|---|---|---|---|---|
| Adaptive launcher icon (AI-node + shield + device concept) | Recognizable brand | S–M | ⬜ | currently default icon; needs a designed adaptive icon (fg/bg layers) |
| Play Store screenshots (5 brief messages) | Store listing | M | ⬜ | design task; drive from the finished screens |
| Inline helper text / "?" tooltips for GGUF, quant, context, LAN | Make jargon approachable | S–M | ⬜ | cross-cutting UX; short localized explanations |

---

## Suggested sequencing

- **Phase 1 — polish the app that exists (mostly S/M):** brand theme (light/dark + palette), bottom
  nav, Markdown, chat actions (copy/regenerate/stop/clear), prompt chips, status messaging, the
  Performance/Privacy settings that map to existing params, onboarding + device-readiness, launcher
  icon, helper tooltips.
- **Phase 2 — the model manager:** managed import library → `GgufInspector` model cards → RAM/compat
  badges → filters → multi-model switch/delete. Then decide on the **download manager** (`INTERNET`).
- **Phase 3 — multimodal:** image (then audio) input with a vision/audio GGUF + mmproj.
- **Phase 4 — hardening/perf:** encrypted conversations, biometric lock, battery-saver + thermal
  throttling, optional Adreno/OpenCL GPU flavor.
- **Phase 5 — the local API server (`XL`, foundational):** in-app Ktor/NanoHTTPD server wrapping the
  JNI core, OpenAI/Ollama routes, API key + LAN toggle + warnings, WebUI, foreground service. This is
  the largest single effort and the brief's flagship differentiator.

> Repo-wide TODOs live in the root [`TODO.md`](../TODO.md); this file scopes only the `android-llmservice` app.
