## Betelgeuse Current Architecture

### Goal

Local-first personalized voice pipeline for your laptop:

- `STT -> LLM -> TTS`
- Session memory across turns
- Auto-turn voice loop with silence-triggered send
- Model selection from the local `models/` folder
- FastAPI backend + lightweight Vite frontend
- Tuned for:
  - `CPU`: Intel i7-13620H
  - `GPU`: NVIDIA RTX 4060 Laptop GPU 8 GB

### Hardware Strategy

- `STT` runs on `CPU`
  - Faster-Whisper `small.en`
  - `compute_type="int8"`
  - `cpu_threads=8`
  - built-in VAD enabled
- `LLM` runs on `GPU`
  - Qwen now has two runtime paths:
    - Transformers with `8-bit` bitsandbytes loading for Hugging Face model folders
    - `llama.cpp` for local GGUF files such as `Q4_K_M` and `Q6_K`
  - reply length and decoding remain manifest-configurable through the hyperparameter editor
- `TTS` runs on `GPU`
  - Available paths: `Melo EN_INDIA` and `Kokoro ONNX`
  - Current verified GPU path on this machine is `Melo EN_INDIA`
  - `Kokoro` currently falls back to CPU here because ONNX Runtime does not expose `CUDAExecutionProvider`

### Current Verified Paths

- Verified:
  - `Qwen3.5-2B-int8` was materialized locally in `models/LLMs/Qwen3.5-2B-int8`
  - LLM int8 GPU generation works
  - `Qwen3.5-4B-Q4_K_M` is now discoverable as a separate LLM option and answers through the local `llama.cpp` runtime
  - `Qwen3.5-4B-Q6_K` is now discoverable as a separate LLM option
  - Melo TTS synthesis works and writes output audio
  - `Qwen3.5-4B` was quantized locally into GGUF artifacts:
    - `models/LLMs/Qwen3.5-4B-Q6_K.gguf`
    - `models/LLMs/Qwen3.5-4B-Q4_K_M.gguf`
  - Full backend pipeline smoke test works:
    - start session
    - auto-ready backend session
    - preserve conversation history
    - generate LLM response
    - synthesize response audio
    - stop session
- Available but not the default GPU path:
  - `Kokoro` assets are downloaded locally and synthesis works
  - in this environment ONNX Runtime exposed CPU provider only, so Kokoro is not the active GPU TTS path

### Project Layout

```text
Betelgeuse/
├── .venv/                         # repo-local Python 3.10 environment
├── backend/
│   └── app/
│       ├── api/
│       │   └── routes.py         # REST API for models, session start/stop, turns, audio
│       ├── core/
│       │   ├── schemas.py        # Pydantic contracts
│       │   └── settings.py       # paths + laptop-tuned defaults
│       ├── pipeline/
│       │   ├── base.py           # swappable STT / LLM / TTS interfaces
│       │   ├── stt/
│       │   │   └── faster_whisper_engine.py
│       │   ├── llm/
│       │   │   ├── llama_cpp_engine.py
│       │   │   └── transformers_engine.py
│       │   └── tts/
│       │       ├── kokoro_engine.py
│       │       └── melo_engine.py
│       ├── services/
│       │   ├── hardware.py       # hardware profile reporting
│       │   ├── model_registry.py # discover models from models/
│       │   ├── pipeline_service.py
│       │   └── session_store.py  # persist session memory as JSON
│       └── main.py               # FastAPI app
├── frontend/
│   ├── src/
│   │   ├── App.tsx               # sidebar + live voice-loop UI
│   │   ├── styles.css
│   │   └── lib/
│   │       ├── api.ts            # typed frontend API client
│   │       └── audioRecorder.ts  # silence-aware browser WAV recorder
│   └── dist/                     # Vite production build
├── models/
│   ├── STTs/
│   │   └── faster-whisper-small-en.json
│   ├── LLMs/
│   │   ├── Qwen3.5-2B/
│   │   └── Qwen3.5-2B-int8/
│   │   ├── Qwen3.5-4B/
│   │   ├── Qwen3.5-4B-Q6_K.gguf
│   │   └── Qwen3.5-4B-Q4_K_M.gguf
│   └── TTSs/
│       ├── kokoro-en-us.json
│       ├── kokoro-v0_19/
│       └── melo-en-india.json
├── runtime/
│   ├── audio/                    # generated wav files
│   └── sessions/                 # persisted session state json
├── scripts/
│   ├── quantize_qwen_int8.py
│   ├── quantize_qwen_gguf.py
│   └── setup_env.sh
└── architecture.md
```

### Model Discovery Design

The UI does not hardcode model choices.

- Backend scans:
  - `models/STTs`
  - `models/LLMs`
  - `models/TTSs`
- Discovery rules:
  - `.json` manifests are loaded directly
  - `manifest.json` inside a model directory is treated as that model's editable local manifest
  - Hugging Face model directories under `models/LLMs` are inferred automatically
  - `.gguf` files under `models/LLMs` are inferred automatically as `llama.cpp` models
  - ONNX TTS folders under `models/TTSs` are inferred automatically

Hyperparameter persistence now follows these rules:

- existing root manifests are edited in place
  - example: `models/STTs/faster-whisper-small-en.json`
  - example: `models/TTSs/melo-en-india.json`
- inferred directory-based models get a local embedded manifest on save
  - example: `models/LLMs/Qwen3.5-2B-int8/manifest.json`
- inferred GGUF models get a local sidecar manifest on save
  - example: `models/LLMs/Qwen3.5-4B-Q4_K_M.json`

This makes each pipeline component replaceable without changing the core app.

### Session Architecture

Each started conversation creates a `session_id`.

For each session we store:

- selected `STT`, `LLM`, `TTS` manifests
- `system_prompt`
- `conversation_history`
- timestamps
- status: `warming | ready | error | closed`

Engine instances are also cached behind the session layer so the pipeline remains component-swappable:

- `STT`, `LLM`, and `TTS` all implement a shared `warmup()` hook
- warmup runs once per engine instance
- this moves first-use latency into the session warm phase instead of the first live user turn
- idle engines are kept hot for `600s` after session stop so reusing the same model stack does not force another cold load

For sidebar navigation we also derive:

- title from the first user message
- preview from the latest message
- whether the session is currently live in memory

Session memory is persisted to:

- `runtime/sessions/<session_id>.json`

So conversation context survives backend lifecycle better than pure in-memory storage.

### Backend Flow

#### 1. Model selection

Frontend loads `/api/models` and shows three dropdowns:

- `STT`
- `LLM`
- `TTS`

Current UI defaults:

- `STT`: Faster-Whisper small.en
- `LLM`: prefer `Q6` or `Q4` GGUF Qwen automatically when present, then fall back to `int8`
- `TTS`: prefer `Melo` automatically so speech synthesis stays on the GPU
- `Silence auto-send`: adjustable in the UI, default `1.0s`
- `Change Hyperparameters`: opens the dedicated configuration page

#### 1A. Hyperparameter editing

Frontend route:

- `#/hyperparameters`

This page lets you tune each pipeline stage separately and also edit the global session behavior:

- `System Prompt`
  - stored in frontend local state and persisted in `localStorage`
  - sent into backend session creation as `system_prompt`
  - lets you change the default assistant behavior without editing code

- `STT`
  - device (`cpu` / `cuda`)
  - compute type
  - CPU threads
  - beam size
  - language
  - recognition prompt
  - VAD settings
- `LLM`
  - generation controls:
    - max new tokens
    - sampling on/off
    - temperature
    - top-p
    - repetition penalty
  - GGUF runtime controls:
    - context size
    - CPU threads
    - batch threads
    - batch size
    - GPU layers
    - flash attention
- `TTS`
  - device (`gpu` / `cpu`)
  - speaker / accent
  - language
  - speed
  - speech style prompt
  - voice / language variant for Kokoro-style engines

Backend routes:

- `GET /api/models/{component}/{model_id}/hyperparameters`
- `PUT /api/models/{component}/{model_id}/hyperparameters`

Saving hyperparameters:

- writes the updated config back to the model manifest
- invalidates the cached engine for that model id
- active live sessions keep using their already loaded engine
- the next session uses the updated hyperparameters automatically

#### 2. Start pipeline

`POST /api/pipeline/start`

Backend:

- resolves selected manifests
- loads or reuses cached engines
- warms `STT -> LLM -> TTS` once before the session becomes ready
- creates session record
- returns ready session metadata

In the new UX, pressing `Start Conversation` does two things:

- starts the backend session
- immediately starts frontend microphone listening

#### 3. Turn processing

Audio turn:

- frontend listens continuously after start/resume
- frontend detects speech locally
- when silence exceeds the configured threshold, the current utterance is closed automatically
- browser sends WAV to `/api/sessions/{id}/turns/audio`
- STT transcribes on CPU
- transcript appended to session memory
- LLM generates response on GPU
- TTS synthesizes WAV
- backend returns timing metrics:
  - `stt_ms`
  - `llm_ms`
  - `tts_ms`
  - `pipeline_ms`
- frontend auto-plays the generated audio
- while processing or speaking, the mic stays locked so the user cannot interrupt with a new question
- after playback ends, frontend automatically resumes listening unless the user paused

#### 4. Stop pipeline

The UI now uses two different stop behaviors:

- `Pause`
  - stops listening immediately
  - aborts any in-flight frontend request
  - stops current output playback
  - does **not** destroy the conversation memory
- `Stop session` backend route: `POST /api/sessions/{id}/stop`
  - used when explicitly ending or replacing the live conversation
  - releases cached references
  - closes session
  - preserves session JSON on disk

#### 5. Sidebar and saved conversations

Backend exposes:

- `GET /api/sessions`
- `GET /api/sessions/{id}`

Frontend uses these to render a ChatGPT-style left sidebar with saved chats.

Selecting an older conversation:

- loads its transcript from persisted session JSON
- shows it in read/view mode
- keeps saved history visible even after the live session is closed

#### 6. Live metrics

Backend exposes:

- `GET /api/metrics/live`

This provides live GPU memory information for the UI.

Frontend displays:

- live VRAM usage
- STT time
- LLM time
- TTS time
- end-to-end delay from `voice heard` to `response spoken`

### TTS Style Prompt

The current TTS stack does not expose a native acoustic prompt in the same way an LLM does.

To make delivery style configurable anyway, the app now supports a manifest-driven `style_prompt` on TTS models:

- editable from the hyperparameters page for both `Kokoro` and `Melo`
- stored in the selected TTS manifest
- applied as hidden speech-delivery guidance when composing the LLM system prompt for each turn

This lets you steer outputs toward tones like:

- warm and reassuring
- calm and professional
- energetic and upbeat
- soft and empathetic

without coupling the feature to one specific TTS engine implementation.

### UI Layout Stability

The main app shell now uses a fixed-height viewport with internal scrolling zones instead of letting the whole page reflow.

- `html`, `body`, and `#root` are pinned to full height
- sidebar and transcript regions scroll internally
- the workspace keeps a stable scrollbar gutter
- this prevents the main window from shifting downward when the app is idle or when playback/listening state changes

### Latency Optimization Layer

This round added a reusable latency-tuning layer that still works if the pipeline components change later.

- `backend/app/pipeline/base.py`
  - all engines now expose a `warmup()` hook
- `backend/app/services/pipeline_service.py`
  - warms engines before a session is marked ready
  - tracks which cached engines are already warmed
  - keeps recently used engines hot across session stop/start within the same backend process
- `backend/app/pipeline/llm/transformers_engine.py`
  - keeps decoding behavior configurable through manifests/settings
  - adds a tiny GPU warmup generation
- `backend/app/pipeline/tts/melo_engine.py`
  - performs a hidden warmup synthesis so the first real spoken reply is not penalized
  - now forces a fully local Hugging Face cache path for Melo assets and BERT tokenizers
  - replaces the stock `tts_to_file()` path with a leaner in-engine synthesis path
  - avoids per-request CUDA cache clearing
  - exposes faster Melo knobs:
    - `disable_bert`
    - `half_precision`
    - `sentence_pause_ms`
- `backend/app/pipeline/stt/faster_whisper_engine.py`
  - performs a tiny silent transcription warmup
- `scripts/profile_pipeline.py`
  - benchmarks the full local pipeline with any `STT / LLM / TTS` combination selected by model id

### Benchmark Snapshot

Public sample used for profiling:

- `runtime/benchmarks/how_old_are_you.wav`
- source: Wikimedia Commons public speech sample

Baseline before this optimization pass:

- `start_session`: `19.7s`
- first live turn:
  - `STT`: `1778.9ms`
  - `LLM`: `3164.1ms`
  - `TTS`: `4981.2ms`
  - `pipeline`: `9924.9ms`
- second warm turn:
  - `STT`: `1744.8ms`
  - `LLM`: `2829.8ms`
  - `TTS`: `181.9ms`
  - `pipeline`: `4757.2ms`

After warmup + engine caching:

- `start_session`: `25.1s`
- first live turn:
  - `STT`: `1832.0ms`
  - `LLM`: `1166.4ms`
  - `TTS`: `92.3ms`
  - `pipeline`: `3091.3ms`
- second warm turn:
  - `STT`: `1644.8ms`
  - `LLM`: `1248.6ms`
  - `TTS`: `105.3ms`
  - `pipeline`: `2999.3ms`

What this means on your laptop right now:

- the steady-state bottleneck is no longer TTS
- TTS cold-start penalty was moved out of the first user reply
- the main remaining live-turn bottleneck is now `STT ~1.6-1.8s`, with `LLM ~1.2s`
- the next remaining cold-path bottleneck is `session start/load time`

### Benchmark Round: User `.mpeg` Prompts

New benchmark audio files used in this round:

- `runtime/benchmarks/.mpeg/make schedule for me.mpeg`
- `runtime/benchmarks/.mpeg/remember my day.mpeg`
- `runtime/benchmarks/.mpeg/story of a cat and dog.mpeg`

Transcripts extracted locally:

- `make schedule for me`
  - `Can you make a comprehensive schedule for me? A schedule for day to day life?`
- `remember my day`
  - `My plans for tomorrow is something like this ... Can you record this or can you keep this in mind?`
- `story of a cat and dog`
  - `Tell me a story about a cat and its relationship with owner and its copate dog`

Important constraint from this round:

- `max_new_tokens` was intentionally left unchanged
- optimization focus was moved to the `TTS` layer instead of shortening LLM output length

TTS benchmark on representative replies shaped from those prompts:

- `Kokoro`
  - schedule-style reply, `332 chars`: `5004.7ms`
  - memory-style reply, `280 chars`: `3107.2ms`
  - story-style reply, `559 chars`: `6666.4ms`
- `Melo`
  - schedule-style reply, `332 chars`: `9176.1ms`
  - memory-style reply, `280 chars`: `10008.5ms`
  - story-style reply, `559 chars`: `14837.6ms`

Practical conclusion for this laptop:

- for short replies, Melo can still feel acceptable
- for medium and long replies, Kokoro is currently the better latency choice
- the app now defaults to `Kokoro` for the speed-first path

Hot restart behavior in the same backend process:

- first `start_session` with the current stack: `25.0s`
- second `start_session` immediately after stop with the same stack: `1.4ms`

So the current runtime strategy is:

- first launch pays the model load + warmup cost once
- conversation turns then stay near `~3.0s` end to end on the tested sample
- stopping and starting the same model stack again stays effectively instant while the backend remains alive

### Swappable Component Design

The app is deliberately split behind interfaces:

- `STTEngine`
- `LLMEngine`
- `TTSEngine`

Current implementations:

- `FasterWhisperEngine`
- `LlamaCppServerEngine`
- `TransformersQwenEngine`
- `KokoroTTSEngine`
- `MeloTTSEngine`

To swap a component later, we only need:

1. new adapter implementation
2. provider mapping in `pipeline_service.py`
3. model manifest or model folder

The new optimization layer stays compatible with that design because the only contract it assumes is:

- `transcribe(...)`
- `generate(...)`
- `synthesize(...)`
- optional `warmup()`

The hyperparameter editor also stays compatible with the swappable design because it is manifest-driven:

- the backend returns field metadata per provider
- the frontend renders those fields generically
- provider-specific defaults live in the backend registry instead of being hardcoded in the UI

### Why This Fits Your Laptop

- Whisper stays on CPU and uses fixed thread budgeting so GPU is reserved for generation and speech synthesis
- Qwen can now run either in Transformers `8-bit` mode or in `llama.cpp` GGUF `Q4/Q6` mode, which fits the 4060 8 GB profile much better than full precision
- TTS defaults to Melo because that is the verified GPU synthesis path on this machine
- Session memory is simple and local, so there is no external DB or cloud dependency increasing latency
- Frontend now uses a silence-aware local recorder, so the interaction feels closer to a live assistant than manual push-to-talk
- Timing and VRAM counters make bottlenecks visible while tuning the pipeline on this exact laptop

### Important Current Notes

- Python environment is local to this repo: `.venv`
- `Qwen3.5-2B-int8` already exists locally
- `Qwen3.5-4B-Q6_K.gguf` and `Qwen3.5-4B-Q4_K_M.gguf` are now saved locally beside the source 4B model
- `Kokoro` model files are already downloaded locally
- `Melo` required extra local setup:
  - linked `unidic-lite` into the expected `unidic` path
  - downloaded small NLTK resources
- The active runtime now supports both Transformers-based Qwen folders and `llama.cpp` GGUF files for Qwen

### Recommended Default Pipeline Right Now

- `STT`: `faster-whisper-small-en`
- `LLM`: `Qwen3.5-4B-Q6_K` for the better speed/quality balance, or `Qwen3.5-4B-Q4_K_M` for a smaller VRAM footprint
- `TTS`: `melo-en-india`
- `Silence auto-send`: `1.0s`
