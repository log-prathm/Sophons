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
  - Qwen loaded through Transformers
  - primary path is `8-bit` quantized loading on the RTX 4060
- `TTS` runs on `GPU`
  - Verified working GPU TTS path: `Melo EN_INDIA`
  - Secondary available path: `Kokoro ONNX`

### Current Verified Paths

- Verified:
  - `Qwen3.5-2B-int8` was materialized locally in `models/LLMs/Qwen3.5-2B-int8`
  - LLM int8 GPU generation works
  - Melo TTS synthesis works and writes output audio
  - Full backend pipeline smoke test works:
    - start session
    - auto-ready backend session
    - preserve conversation history
    - generate LLM response
    - synthesize response audio
    - stop session
- Available but not the default GPU path:
  - `Kokoro` assets are downloaded locally and synthesis works
  - in this environment ONNX Runtime exposed CPU provider only, so Kokoro should be treated as the lighter alternate TTS path, not the primary verified GPU path

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
│   └── TTSs/
│       ├── kokoro-en-us.json
│       ├── kokoro-v0_19/
│       └── melo-en-india.json
├── runtime/
│   ├── audio/                    # generated wav files
│   └── sessions/                 # persisted session state json
├── scripts/
│   ├── quantize_qwen_int8.py
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
  - Hugging Face model directories under `models/LLMs` are inferred automatically
  - ONNX TTS folders under `models/TTSs` are inferred automatically

This makes each pipeline component replaceable without changing the core app.

### Session Architecture

Each started conversation creates a `session_id`.

For each session we store:

- selected `STT`, `LLM`, `TTS` manifests
- `system_prompt`
- `conversation_history`
- timestamps
- status: `warming | ready | error | closed`

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
- `LLM`: prefer `int8` Qwen automatically
- `TTS`: prefer `Melo` automatically
- `Silence auto-send`: adjustable in the UI, default `1.0s`

#### 2. Start pipeline

`POST /api/pipeline/start`

Backend:

- resolves selected manifests
- loads or reuses cached engines
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

### Swappable Component Design

The app is deliberately split behind interfaces:

- `STTEngine`
- `LLMEngine`
- `TTSEngine`

Current implementations:

- `FasterWhisperEngine`
- `TransformersQwenEngine`
- `KokoroTTSEngine`
- `MeloTTSEngine`

To swap a component later, we only need:

1. new adapter implementation
2. provider mapping in `pipeline_service.py`
3. model manifest or model folder

### Why This Fits Your Laptop

- Whisper stays on CPU and uses fixed thread budgeting so GPU is reserved for generation and speech synthesis
- Qwen is quantized to `8-bit`, which fits the 4060 8 GB profile much better than full precision
- TTS defaults to Melo because that is the verified GPU route on this machine
- Session memory is simple and local, so there is no external DB or cloud dependency increasing latency
- Frontend now uses a silence-aware local recorder, so the interaction feels closer to a live assistant than manual push-to-talk
- Timing and VRAM counters make bottlenecks visible while tuning the pipeline on this exact laptop

### Important Current Notes

- Python environment is local to this repo: `.venv`
- `Qwen3.5-2B-int8` already exists locally
- `Kokoro` model files are already downloaded locally
- `Melo` required extra local setup:
  - linked `unidic-lite` into the expected `unidic` path
  - downloaded small NLTK resources
- Because of that, Melo should be considered the primary working GPU TTS path right now

### Recommended Default Pipeline Right Now

- `STT`: `faster-whisper-small-en`
- `LLM`: `Qwen3.5-2B-int8`
- `TTS`: `melo-en-india`
- `Silence auto-send`: `1.0s`
