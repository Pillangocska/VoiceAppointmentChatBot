import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { connectSession, type Connection } from "./api";
import { PushToTalkRecorder, base64ToWavUrl, blobToBase64 } from "./recorder";
import type {
  BookingPayload,
  DomainPayload,
  Lang,
  SentimentPayload,
  ServerEvent,
  TranscriptTurn,
} from "./types";
import "./App.css";

type Status =
  | "connecting"
  | "ready"
  | "recording"
  | "processing"
  | "complete"
  | "error";

interface SavedPayload {
  path: string | null;
  payload: Record<string, unknown>;
}

const SENTIMENT_EMOJI: Record<string, string> = {
  positive: "😊",
  neutral: "😐",
  negative: "😟",
};

export default function App() {
  const [status, setStatus] = useState<Status>("connecting");
  const [domain, setDomain] = useState<DomainPayload | null>(null);
  const [booking, setBooking] = useState<BookingPayload | null>(null);
  const [turns, setTurns] = useState<TranscriptTurn[]>([]);
  const [sentiment, setSentiment] = useState<SentimentPayload | null>(null);
  const [saved, setSaved] = useState<SavedPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [latestLang, setLatestLang] = useState<Lang>("en");

  const recorderRef = useRef<PushToTalkRecorder | null>(null);
  const connectionRef = useRef<Connection | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const chatRef = useRef<HTMLDivElement | null>(null);

  const handleEvent = useCallback((event: ServerEvent) => {
    switch (event.type) {
      case "session_ready":
        setDomain(event.domain);
        setBooking(event.booking);
        setStatus("ready");
        break;
      case "transcript":
        if (event.text.trim()) {
          if (event.language === "en" || event.language === "hu") {
            setLatestLang(event.language);
          }
          setTurns((prev) => [
            ...prev,
            { role: "user", text: event.text, language: event.language },
          ]);
        }
        break;
      case "sentiment":
        setSentiment({ label: event.label, score: event.score });
        setTurns((prev) => {
          if (prev.length === 0) return prev;
          const last = prev[prev.length - 1];
          if (last.role !== "user") return prev;
          const updated = [...prev];
          updated[updated.length - 1] = {
            ...last,
            sentiment: { label: event.label, score: event.score },
          };
          return updated;
        });
        break;
      case "reply":
        setBooking(event.booking);
        if (event.language === "en" || event.language === "hu") {
          setLatestLang(event.language);
        }
        setTurns((prev) => [
          ...prev,
          {
            role: "assistant",
            text: event.text,
            language: event.language,
          },
        ]);
        if (event.audio_base64 && audioRef.current) {
          const url = base64ToWavUrl(event.audio_base64);
          audioRef.current.src = url;
          audioRef.current.play().catch((err) => {
            console.warn("audio playback failed", err);
          });
        }
        setStatus(event.booking_complete ? "complete" : "ready");
        break;
      case "saved":
        setSaved({ path: event.path, payload: event.payload });
        setStatus("complete");
        break;
      case "error":
        setError(event.message);
        setStatus("error");
        break;
    }
  }, []);

  useEffect(() => {
    const conn = connectSession(
      handleEvent,
      (err) => {
        console.error("websocket error", err);
        setError("WebSocket connection error");
        setStatus("error");
      },
      () => {
        if (status !== "complete") {
          setStatus((prev) => (prev === "complete" ? prev : "error"));
        }
      },
    );
    connectionRef.current = conn;
    return () => conn.close();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [handleEvent]);

  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [turns]);

  const hotkeyHeldRef = useRef(false);

  const startRecording = useCallback(async () => {
    if (status !== "ready") return;
    setError(null);
    try {
      const recorder = new PushToTalkRecorder();
      await recorder.start();
      recorderRef.current = recorder;
      setStatus("recording");
    } catch (err) {
      console.error(err);
      setError("Microphone access denied or unavailable");
      setStatus("ready");
    }
  }, [status]);

  const stopRecording = useCallback(async () => {
    const recorder = recorderRef.current;
    if (!recorder) return;
    recorderRef.current = null;
    setStatus("processing");
    try {
      const blob = await recorder.stop();
      if (!blob) {
        setStatus("ready");
        return;
      }
      const b64 = await blobToBase64(blob);
      connectionRef.current?.send({
        type: "utterance",
        mime: blob.type,
        audio_base64: b64,
      });
    } catch (err) {
      console.error(err);
      setError("Recording failed");
      setStatus("ready");
    }
  }, []);

  useEffect(() => {
    const isTypingTarget = (target: EventTarget | null): boolean => {
      if (!(target instanceof HTMLElement)) return false;
      const tag = target.tagName;
      return (
        tag === "INPUT" ||
        tag === "TEXTAREA" ||
        tag === "SELECT" ||
        target.isContentEditable
      );
    };

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key !== "Enter") return;
      if (e.repeat) return;
      if (isTypingTarget(e.target)) return;
      if (status !== "ready") return;
      e.preventDefault();
      hotkeyHeldRef.current = true;
      startRecording();
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.key !== "Enter") return;
      if (!hotkeyHeldRef.current) return;
      hotkeyHeldRef.current = false;
      e.preventDefault();
      stopRecording();
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, [status, startRecording, stopRecording]);

  const downloadJson = useCallback(() => {
    if (!saved) return;
    const blob = new Blob([JSON.stringify(saved.payload, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `appointment_${domain?.name ?? "booking"}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }, [saved, domain]);

  const statusLabel = useMemo(() => {
    switch (status) {
      case "connecting": return "Connecting";
      case "ready":      return "Ready";
      case "recording":  return "Listening";
      case "processing": return "Thinking";
      case "complete":   return "Complete";
      case "error":      return "Error";
    }
  }, [status]);

  const statusDotClass = useMemo(() => {
    if (status === "ready" || status === "complete") return "connected";
    if (status === "recording") return "recording";
    if (status === "processing") return "processing";
    return "";
  }, [status]);

  const totalSlots = domain?.slots.length ?? 0;
  const completedSlots = useMemo(() => {
    if (!domain || !booking) return 0;
    return domain.slots.filter((spec) => {
      const value = booking.slots[spec.name] ?? "";
      const confirmed = booking.confirmed.includes(spec.name);
      const pendingPhone =
        spec.type === "phone" &&
        booking.pending_phone_confirmation &&
        !!value &&
        !confirmed;
      return confirmed || (!!value && !pendingPhone);
    }).length;
  }, [domain, booking]);

  const progressPct =
    totalSlots === 0 ? 0 : Math.round((completedSlots / totalSlots) * 100);

  return (
    <div className="app">
      <header className="header glass">
        <div className="header-brand">
          <div className="brand-mark" aria-hidden="true">
            <SparklesIcon />
          </div>
          <div className="brand-text">
            <h1 className="brand-title">Voice Booking</h1>
            <div className="brand-subtitle">
              {domain ? domain.display_name[latestLang] : "Loading…"}
              <span className="lang-chip">{latestLang}</span>
            </div>
          </div>
        </div>

        <div className="header-control">
          <div className="header-control-top">
            <div className="ptt-side">
              <span className="ptt-label">{pttLabel(status)}</span>
              <p className="ptt-hint">
                Hold the mic <kbd className="kbd">or Enter</kbd> to record,
                release to send. Replies match your language — English or
                Magyar.
              </p>
            </div>
            <PushToTalkButton
              status={status}
              onStart={startRecording}
              onStop={stopRecording}
            />
          </div>
          <div className="status-pill">
            <span className={`status-dot ${statusDotClass}`} />
            {statusLabel}
          </div>
        </div>
      </header>

      <main className="chat glass" ref={chatRef}>
        {error && (
          <div className="error-banner">
            <AlertIcon />
            {error}
          </div>
        )}
        {turns.length === 0 && status === "ready" && (
          <div className="empty-state">
            <div className="empty-icon" aria-hidden="true">
              <MicIcon />
            </div>
            <h3>Ready when you are</h3>
            <p>
              Press and hold the mic, say something to book your appointment,
              then release. I'll fill in the details as you speak.
            </p>
            <div className="empty-langs">
              <span className="lang-chip">English</span>
              <span className="lang-chip">Magyar</span>
            </div>
          </div>
        )}
        {turns.map((turn, idx) => (
          <ChatBubble key={idx} turn={turn} />
        ))}
        <audio ref={audioRef} hidden />
      </main>

      <aside className="panel glass">
        <section className="panel-section">
          <h2>
            Appointment details
            <span className="count-chip">
              {completedSlots}/{totalSlots}
            </span>
          </h2>

          <div className="progress">
            <div className="progress-track">
              <div
                className="progress-fill"
                style={{ width: `${progressPct}%` }}
              />
            </div>
            <div className="progress-label">
              <span>{progressPct}% complete</span>
              <strong>
                {booking?.is_complete ? "All set" : "In progress"}
              </strong>
            </div>
          </div>

          <div className="slot-list">
            {domain?.slots.map((spec) => {
              const value = booking?.slots[spec.name] ?? "";
              const confirmed =
                booking?.confirmed.includes(spec.name) ?? false;
              const pending =
                spec.type === "phone" &&
                booking?.pending_phone_confirmation &&
                !!value &&
                !confirmed;
              const state: SlotState = confirmed
                ? "confirmed"
                : pending
                  ? "pending"
                  : value
                    ? "captured"
                    : "missing";
              return (
                <div className={`slot ${state}`} key={spec.name}>
                  <div className="slot-header">
                    <span className="slot-name">{spec.name}</span>
                    <span className="slot-status-icon">
                      <SlotStateIcon state={state} />
                    </span>
                  </div>
                  <div className={`slot-value ${value ? "" : "empty"}`}>
                    {value || spec.prompt[latestLang]}
                  </div>
                  <div className="slot-status-text">
                    {slotStateLabel(state)}
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        {sentiment && (
          <section className="panel-section">
            <h2>Sentiment</h2>
            <div className="sentiment-card">
              <div className="sentiment-label">
                <span className="sentiment-emoji">
                  {SENTIMENT_EMOJI[sentiment.label] ?? "🙂"}
                </span>
                <span className="sentiment-text">{sentiment.label}</span>
              </div>
              <span className="sentiment-score">
                {(sentiment.score * 100).toFixed(0)}%
              </span>
            </div>
          </section>
        )}

        {saved && (
          <section className="panel-section">
            <h2>Saved</h2>
            <div className="saved-card">
              <p className="saved-path">
                {saved.path ?? "(no server path)"}
              </p>
              <button className="download-btn" onClick={downloadJson}>
                <DownloadIcon />
                Download JSON
              </button>
            </div>
          </section>
        )}
      </aside>
    </div>
  );
}

type SlotState = "confirmed" | "pending" | "captured" | "missing";

function slotStateLabel(state: SlotState): string {
  switch (state) {
    case "confirmed": return "Confirmed";
    case "pending":   return "Awaiting confirmation";
    case "captured":  return "Captured";
    case "missing":   return "Waiting…";
  }
}

function pttLabel(status: Status): string {
  switch (status) {
    case "connecting": return "Connecting to assistant…";
    case "ready":      return "Hold to speak";
    case "recording":  return "Listening to you";
    case "processing": return "Thinking it over";
    case "complete":   return "Booking complete";
    case "error":      return "Something went wrong";
  }
}

function ChatBubble({ turn }: { turn: TranscriptTurn }) {
  return (
    <div className={`bubble-row ${turn.role}`}>
      <div className={`bubble ${turn.role}`}>{turn.text}</div>
      <div className="bubble-meta">
        <span className="tag">{turn.language}</span>
        {turn.sentiment && (
          <span className={`tag ${turn.sentiment.label}`}>
            {turn.sentiment.label} · {(turn.sentiment.score * 100).toFixed(0)}%
          </span>
        )}
      </div>
    </div>
  );
}

interface PushToTalkButtonProps {
  status: Status;
  onStart: () => void;
  onStop: () => void;
}

function PushToTalkButton({ status, onStart, onStop }: PushToTalkButtonProps) {
  const disabled = status !== "ready" && status !== "recording";
  const recording = status === "recording";
  const processing = status === "processing";

  const handlePointerDown = (event: React.PointerEvent<HTMLButtonElement>) => {
    event.preventDefault();
    if (status === "ready") onStart();
  };
  const handlePointerUp = (event: React.PointerEvent<HTMLButtonElement>) => {
    event.preventDefault();
    if (status === "recording") onStop();
  };

  return (
    <div className={`ptt-wrap ${recording ? "recording" : ""}`}>
      <span className="ptt-ring" aria-hidden="true" />
      <span className="ptt-ring" aria-hidden="true" />
      <span className="ptt-ring" aria-hidden="true" />
      <button
        className={`ptt-button ${recording ? "recording" : ""}`}
        onPointerDown={handlePointerDown}
        onPointerUp={handlePointerUp}
        onPointerLeave={(e) => status === "recording" && handlePointerUp(e)}
        disabled={disabled}
        aria-label={recording ? "Release to send" : "Hold to record"}
      >
        {recording ? (
          <span className="ptt-waveform" aria-hidden="true">
            <span /><span /><span /><span /><span />
          </span>
        ) : processing ? (
          <span className="ptt-thinking" aria-hidden="true">
            <span /><span /><span />
          </span>
        ) : (
          <MicIcon />
        )}
      </button>
    </div>
  );
}

/* ====== Inline icons (no extra deps) ====== */

function MicIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
      strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z" />
      <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
      <line x1="12" y1="19" x2="12" y2="23" />
      <line x1="8" y1="23" x2="16" y2="23" />
    </svg>
  );
}

function SparklesIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
      strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 3l1.6 4.4L18 9l-4.4 1.6L12 15l-1.6-4.4L6 9l4.4-1.6L12 3z" />
      <path d="M19 14l.8 2.2L22 17l-2.2.8L19 20l-.8-2.2L16 17l2.2-.8L19 14z" />
    </svg>
  );
}

function DownloadIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
      strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="7 10 12 15 17 10" />
      <line x1="12" y1="15" x2="12" y2="3" />
    </svg>
  );
}

function AlertIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
      strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="8" x2="12" y2="13" />
      <line x1="12" y1="16" x2="12" y2="16" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3"
      strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

function ClockIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"
      strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="9" />
      <polyline points="12 7 12 12 15 14" />
    </svg>
  );
}

function DotIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor">
      <circle cx="12" cy="12" r="4" />
    </svg>
  );
}

function DashIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"
      strokeLinecap="round">
      <line x1="6" y1="12" x2="18" y2="12" />
    </svg>
  );
}

function SlotStateIcon({ state }: { state: SlotState }) {
  switch (state) {
    case "confirmed": return <CheckIcon />;
    case "pending":   return <ClockIcon />;
    case "captured":  return <DotIcon />;
    case "missing":   return <DashIcon />;
  }
}
