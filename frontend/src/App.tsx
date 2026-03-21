import { useEffect, useRef, useState } from "react";

import {
  fetchLiveMetrics,
  fetchModels,
  fetchSession,
  fetchSessions,
  getApiBase,
  LiveMetrics,
  ModelCatalog,
  sendAudioTurn,
  SessionListItem,
  SessionState,
  startPipeline,
  stopPipeline,
  TurnMetrics,
  TurnResponse,
} from "./lib/api";
import { AutoSilenceRecorder, RecordingResult } from "./lib/audioRecorder";

const DEFAULT_SYSTEM_PROMPT =
  "You are a concise, warm local voice assistant running on a personal laptop. Answer clearly, keep latency low, and preserve useful conversational context across turns.";

type VoiceMode =
  | "idle"
  | "starting"
  | "listening"
  | "processing"
  | "speaking"
  | "paused"
  | "viewing";

export default function App() {
  const recorderRef = useRef<AutoSilenceRecorder | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const pendingTurnControllerRef = useRef<AbortController | null>(null);
  const sessionRef = useRef<SessionState | null>(null);
  const modeRef = useRef<VoiceMode>("idle");

  const [catalog, setCatalog] = useState<ModelCatalog | null>(null);
  const [conversations, setConversations] = useState<SessionListItem[]>([]);
  const [selectedStt, setSelectedStt] = useState("");
  const [selectedLlm, setSelectedLlm] = useState("");
  const [selectedTts, setSelectedTts] = useState("");
  const [systemPrompt, setSystemPrompt] = useState(DEFAULT_SYSTEM_PROMPT);
  const [session, setSession] = useState<SessionState | null>(null);
  const [mode, setMode] = useState<VoiceMode>("idle");
  const [silenceSeconds, setSilenceSeconds] = useState<number>(() => {
    const stored = window.localStorage.getItem("betelgeuse.silenceSeconds");
    return stored ? Number(stored) : 1;
  });
  const [micLevel, setMicLevel] = useState(0);
  const [liveMetrics, setLiveMetrics] = useState<LiveMetrics | null>(null);
  const [turnMetrics, setTurnMetrics] = useState<TurnMetrics | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    sessionRef.current = session;
  }, [session]);

  useEffect(() => {
    modeRef.current = mode;
  }, [mode]);

  useEffect(() => {
    void Promise.all([loadCatalog(), loadConversations(), refreshLiveMetrics()]);
  }, []);

  useEffect(() => {
    window.localStorage.setItem("betelgeuse.silenceSeconds", String(silenceSeconds));
  }, [silenceSeconds]);

  useEffect(() => {
    const interval = window.setInterval(() => {
      void refreshLiveMetrics();
    }, 1500);
    return () => window.clearInterval(interval);
  }, []);

  useEffect(() => {
    return () => {
      void pauseConversation(false);
    };
  }, []);

  async function loadCatalog() {
    try {
      const nextCatalog = await fetchModels();
      setCatalog(nextCatalog);
      setSelectedStt((current) => current || nextCatalog.stt[0]?.id || "");
      setSelectedLlm(
        (current) =>
          current ||
          nextCatalog.llm.find((item) => item.id.toLowerCase().includes("int8"))?.id ||
          nextCatalog.llm[0]?.id ||
          "",
      );
      setSelectedTts(
        (current) =>
          current ||
          nextCatalog.tts.find((item) => item.id.toLowerCase().includes("melo"))?.id ||
          nextCatalog.tts[0]?.id ||
          "",
      );
      setError("");
    } catch (nextError) {
      setError((nextError as Error).message);
    }
  }

  async function loadConversations() {
    try {
      const items = await fetchSessions();
      setConversations(items);
    } catch (nextError) {
      setError((nextError as Error).message);
    }
  }

  async function refreshLiveMetrics() {
    try {
      const metrics = await fetchLiveMetrics();
      setLiveMetrics(metrics);
    } catch {
      // Keep the last successful metrics if polling fails.
    }
  }

  async function closeCurrentSessionIfNeeded() {
    const current = sessionRef.current;
    if (!current || current.status === "closed") {
      return;
    }
    await pauseConversation(false);
    try {
      const summary = await stopPipeline(current.session_id);
      setSession(summary.session);
    } catch {
      // If the backend session is already gone, we still want the UI to continue.
    }
  }

  async function handleNewConversation() {
    await closeCurrentSessionIfNeeded();
    setSession(null);
    setTurnMetrics(null);
    setMode("idle");
    setMicLevel(0);
    setError("");
    await loadConversations();
  }

  async function handleStartOrResume() {
    if (mode === "paused" && session?.status === "ready") {
      await beginListening(session);
      return;
    }

    if (!selectedStt || !selectedLlm || !selectedTts) {
      setError("Select one STT, LLM, and TTS model before starting.");
      return;
    }

    try {
      await closeCurrentSessionIfNeeded();
      setMode("starting");
      const response = await startPipeline({
        stt_model_id: selectedStt,
        llm_model_id: selectedLlm,
        tts_model_id: selectedTts,
        system_prompt: systemPrompt,
      });
      setSession(response.session);
      setTurnMetrics(null);
      setError("");
      await loadConversations();
      await beginListening(response.session);
    } catch (nextError) {
      setMode("idle");
      setError((nextError as Error).message);
    }
  }

  async function handleSelectConversation(item: SessionListItem) {
    if (sessionRef.current?.session_id === item.session_id) {
      return;
    }

    await closeCurrentSessionIfNeeded();
    try {
      const summary = await fetchSession(item.session_id);
      setSession(summary.session);
      setTurnMetrics(null);
      setMode("viewing");
      setMicLevel(0);
      setError("");
    } catch (nextError) {
      setError((nextError as Error).message);
    }
  }

  async function beginListening(nextSession: SessionState) {
    await stopRecorder();
    stopAudioPlayback();

    const recorder = new AutoSilenceRecorder();
    recorderRef.current = recorder;
    setMode("listening");
    setMicLevel(0);

    try {
      await recorder.start({
        silenceDurationMs: silenceSeconds * 1000,
        silenceThreshold: 0.016,
        minSpeechMs: 250,
        onLevel: (level) => setMicLevel(Math.min(level * 24, 1)),
        onListeningStarted: () => {
          setMode("listening");
          setError("");
        },
        onSegmentReady: async (segment) => {
          await submitSegment(nextSession.session_id, segment);
        },
      });
    } catch (nextError) {
      setMode("paused");
      setError((nextError as Error).message);
    }
  }

  async function submitSegment(sessionId: string, segment: RecordingResult) {
    setMode("processing");
    setMicLevel(0);
    const controller = new AbortController();
    pendingTurnControllerRef.current = controller;

    try {
      const response = await sendAudioTurn(sessionId, segment.blob, controller.signal);
      pendingTurnControllerRef.current = null;
      applyTurnResponse(response);
      await loadConversations();
      await playAssistantAudio(response, segment.speechStartedAtMs);
    } catch (nextError) {
      pendingTurnControllerRef.current = null;
      if ((nextError as Error).name === "AbortError") {
        setMode("paused");
        return;
      }
      setMode("paused");
      setError((nextError as Error).message);
    }
  }

  function applyTurnResponse(response: TurnResponse) {
    setSession((current) =>
      current
        ? {
            ...current,
            conversation_history: response.conversation_history,
            updated_at: new Date().toISOString(),
          }
        : current,
    );
    setTurnMetrics(response.metrics);
  }

  async function playAssistantAudio(response: TurnResponse, heardAtMs: number) {
    stopAudioPlayback();
    const audio = new Audio(`${getApiBase()}${response.assistant_audio_url}`);
    audioRef.current = audio;

    audio.addEventListener(
      "playing",
      () => {
        setMode("speaking");
        setTurnMetrics((current) =>
          current
            ? {
                ...current,
                heard_to_response_ms: Number((performance.now() - heardAtMs).toFixed(1)),
              }
            : current,
        );
      },
      { once: true },
    );

    audio.addEventListener(
      "ended",
      () => {
        audioRef.current = null;
        if (modeRef.current !== "paused" && sessionRef.current?.status === "ready") {
          void beginListening(sessionRef.current);
        }
      },
      { once: true },
    );

    try {
      await audio.play();
    } catch (nextError) {
      setMode("paused");
      setError((nextError as Error).message);
    }
  }

  async function pauseConversation(keepPausedState = true) {
    await stopRecorder();
    stopAudioPlayback();
    pendingTurnControllerRef.current?.abort();
    pendingTurnControllerRef.current = null;
    setMicLevel(0);
    setMode(keepPausedState && sessionRef.current?.status === "ready" ? "paused" : "idle");
  }

  async function stopRecorder() {
    if (recorderRef.current) {
      const active = recorderRef.current;
      recorderRef.current = null;
      await active.cancel();
    }
  }

  function stopAudioPlayback() {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      audioRef.current.src = "";
      audioRef.current = null;
    }
  }

  const activeConversationId = session?.session_id ?? "";
  const canConfigure = mode === "idle" || mode === "viewing" || session?.status === "closed";
  const showResume = mode === "paused" && session?.status === "ready";
  const showStart = !showResume;

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-header">
          <div>
            <p className="eyebrow">Betelgeuse</p>
            <h1>Chats</h1>
          </div>
          <button className="primary-button" onClick={() => void handleNewConversation()}>
            New Chat
          </button>
        </div>

        <div className="conversation-list">
          {conversations.length ? (
            conversations.map((item) => (
              <button
                key={item.session_id}
                className={`conversation-item ${
                  activeConversationId === item.session_id ? "active" : ""
                }`}
                onClick={() => void handleSelectConversation(item)}
              >
                <strong>{item.title}</strong>
                <span>{item.preview}</span>
                <small>
                  {new Date(item.updated_at).toLocaleString()}
                  {item.is_live ? " · live" : ""}
                </small>
              </button>
            ))
          ) : (
            <div className="conversation-empty">No saved conversations yet.</div>
          )}
        </div>
      </aside>

      <section className="workspace">
        <header className="hero">
          <div>
            <p className="eyebrow">Auto Voice Loop</p>
            <h2>Silence-aware local conversation</h2>
            <p className="hero-copy">
              Press start once. The app listens, detects silence, sends the turn,
              speaks the reply automatically, and resumes listening when output finishes.
            </p>
          </div>
          <div className="hero-status">
            <span className={`status-chip ${mode}`}>{statusLabel(mode, session)}</span>
            <span className="status-note">
              {session ? `Session ${session.session_id.slice(0, 8)}` : "No active session"}
            </span>
          </div>
        </header>

        <section className="top-grid">
          <article className="panel setup-panel">
            <header className="panel-header">
              <div>
                <p className="panel-title">Conversation Controls</p>
                <p className="panel-subtitle">
                  Start begins continuous listening. Pause stops listening and cuts output.
                </p>
              </div>
            </header>

            <div className="field-grid">
              <label className="field">
                <span>STT</span>
                <select
                  value={selectedStt}
                  onChange={(event) => setSelectedStt(event.target.value)}
                  disabled={!canConfigure}
                >
                  {catalog?.stt.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.label}
                    </option>
                  ))}
                </select>
              </label>

              <label className="field">
                <span>LLM</span>
                <select
                  value={selectedLlm}
                  onChange={(event) => setSelectedLlm(event.target.value)}
                  disabled={!canConfigure}
                >
                  {catalog?.llm.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.label}
                    </option>
                  ))}
                </select>
              </label>

              <label className="field">
                <span>TTS</span>
                <select
                  value={selectedTts}
                  onChange={(event) => setSelectedTts(event.target.value)}
                  disabled={!canConfigure}
                >
                  {catalog?.tts.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.label}
                    </option>
                  ))}
                </select>
              </label>

              <label className="field">
                <span>Silence Auto-Send</span>
                <div className="range-row">
                  <input
                    type="range"
                    min="0.6"
                    max="3"
                    step="0.1"
                    value={silenceSeconds}
                    onChange={(event) => setSilenceSeconds(Number(event.target.value))}
                  />
                  <strong>{silenceSeconds.toFixed(1)}s</strong>
                </div>
              </label>
            </div>

            <label className="field">
              <span>System Prompt</span>
              <textarea
                rows={3}
                value={systemPrompt}
                onChange={(event) => setSystemPrompt(event.target.value)}
                disabled={!canConfigure}
              />
            </label>

            <div className="button-row">
              <button className="primary-button" onClick={() => void handleStartOrResume()}>
                {showResume ? "Resume Listening" : "Start Conversation"}
              </button>
              <button
                className="ghost-button"
                onClick={() => void pauseConversation(true)}
                disabled={!session || session.status !== "ready" || mode === "idle" || mode === "viewing"}
              >
                Pause
              </button>
            </div>

            <div className="meter-block">
              <div className="meter-label">
                <span>Mic level</span>
                <strong>{Math.round(micLevel * 100)}%</strong>
              </div>
              <div className="meter-track">
                <div className="meter-fill" style={{ width: `${Math.max(micLevel * 100, 4)}%` }} />
              </div>
            </div>
          </article>

          <article className="panel metrics-panel">
            <header className="panel-header">
              <div>
                <p className="panel-title">Live Performance</p>
                <p className="panel-subtitle">
                  Watch where the delay lives while the pipeline is running.
                </p>
              </div>
            </header>

            <div className="metric-grid">
              <MetricCard
                label="VRAM"
                value={
                  liveMetrics?.gpu_memory_used_mb && liveMetrics.gpu_memory_total_mb
                    ? `${liveMetrics.gpu_memory_used_mb}/${liveMetrics.gpu_memory_total_mb} MB`
                    : "Unavailable"
                }
                secondary={
                  liveMetrics?.gpu_memory_percent != null
                    ? `${liveMetrics.gpu_memory_percent}% used`
                    : liveMetrics?.gpu_name ?? "Waiting for metrics"
                }
              />
              <MetricCard
                label="Delay"
                value={formatMs(turnMetrics?.heard_to_response_ms)}
                secondary="voice heard -> response spoken"
              />
              <MetricCard label="STT" value={formatMs(turnMetrics?.stt_ms)} secondary="CPU transcription" />
              <MetricCard label="LLM" value={formatMs(turnMetrics?.llm_ms)} secondary="GPU generation" />
              <MetricCard label="TTS" value={formatMs(turnMetrics?.tts_ms)} secondary="speech synthesis" />
            </div>

            <div className="notes">
              {catalog?.hardware.notes.map((note) => (
                <p key={note}>{note}</p>
              ))}
            </div>
          </article>
        </section>

        <article className="panel transcript-panel">
          <header className="panel-header">
            <div>
              <p className="panel-title">Conversation</p>
              <p className="panel-subtitle">
                Output is spoken automatically. While processing or speaking, the mic stays locked.
              </p>
            </div>
          </header>

          <div className="session-strip">
            <span>{session?.stt_model.label ?? "No STT selected"}</span>
            <span>{session?.llm_model.label ?? "No LLM selected"}</span>
            <span>{session?.tts_model.label ?? "No TTS selected"}</span>
            <span>{session ? `Updated ${formatTime(session.updated_at)}` : "Waiting to start"}</span>
          </div>

          <div className="messages">
            {session?.conversation_history.length ? (
              session.conversation_history.map((message, index) => (
                <div key={`${message.created_at}-${index}`} className={`message ${message.role}`}>
                  <span className="message-role">{message.role}</span>
                  <p>{message.content}</p>
                </div>
              ))
            ) : (
              <div className="empty-state">
                <p>
                  Press <strong>Start Conversation</strong>, speak normally, and pause after your
                  sentence. Silence handles the send for you.
                </p>
              </div>
            )}
          </div>
        </article>

        {error && <aside className="error-banner">{error}</aside>}
      </section>
    </main>
  );
}

function MetricCard(props: { label: string; value: string; secondary: string }) {
  return (
    <div className="metric-card">
      <span>{props.label}</span>
      <strong>{props.value}</strong>
      <small>{props.secondary}</small>
    </div>
  );
}

function statusLabel(mode: VoiceMode, session: SessionState | null) {
  if (!session) {
    return mode === "starting" ? "starting" : "idle";
  }

  if (mode === "viewing") {
    return "viewing saved chat";
  }

  return mode;
}

function formatMs(value: number | null | undefined) {
  if (value == null) {
    return "—";
  }
  return `${Math.round(value)} ms`;
}

function formatTime(value: string) {
  return new Date(value).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}
