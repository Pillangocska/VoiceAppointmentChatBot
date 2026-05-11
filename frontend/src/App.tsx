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
      case "connecting": return "Connecting…";
      case "ready":      return "Ready";
      case "recording":  return "Recording";
      case "processing": return "Thinking…";
      case "complete":   return "Booking complete";
      case "error":      return "Error";
    }
  }, [status]);

  const statusDotClass = useMemo(() => {
    if (status === "ready" || status === "complete") return "connected";
    if (status === "recording") return "recording";
    if (status === "processing") return "processing";
    return "";
  }, [status]);

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>Voice Booking</h1>
          <div className="domain-name">
            {domain ? domain.display_name[latestLang] : "loading…"}
          </div>
        </div>
        <div className="status-pill">
          <span className={`status-dot ${statusDotClass}`} />
          {statusLabel}
        </div>
      </header>

      <main className="chat" ref={chatRef}>
        {error && <div className="error-banner">{error}</div>}
        {turns.length === 0 && status === "ready" && (
          <div className="bubble assistant">
            Press and hold the button below, say something to book an
            appointment, then release. EN and HU both work.
          </div>
        )}
        {turns.map((turn, idx) => (
          <ChatBubble key={idx} turn={turn} />
        ))}
        <audio ref={audioRef} hidden />
      </main>

      <aside className="panel">
        <section>
          <h2>Booking slots</h2>
          <div className="slot-list">
            {domain?.slots.map((spec) => {
              const value = booking?.slots[spec.name] ?? "";
              const confirmed = booking?.confirmed.includes(spec.name) ?? false;
              const pending =
                spec.type === "phone" &&
                booking?.pending_phone_confirmation &&
                !!value &&
                !confirmed;
              const cls = confirmed
                ? "slot confirmed"
                : pending
                  ? "slot pending"
                  : "slot";
              return (
                <div className={cls} key={spec.name}>
                  <div className="slot-name">{spec.name}</div>
                  <div className="slot-value">
                    {value || spec.prompt[latestLang]}
                  </div>
                  <div
                    className={`slot-status ${
                      confirmed
                        ? "confirmed"
                        : pending
                          ? "pending"
                          : "missing"
                    }`}
                  >
                    {confirmed
                      ? "confirmed"
                      : pending
                        ? "awaiting confirmation"
                        : value
                          ? "captured"
                          : "missing"}
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        {sentiment && (
          <section>
            <h2>Sentiment</h2>
            <div className="sentiment-summary">
              <span className={`tag ${sentiment.label}`}>{sentiment.label}</span>
              <span>{(sentiment.score * 100).toFixed(0)}%</span>
            </div>
          </section>
        )}

        {saved && (
          <section>
            <h2>Appointment saved</h2>
            <p style={{ fontSize: 12, color: "#9fb0d1", margin: "0 0 8px" }}>
              {saved.path ?? "(no server path)"}
            </p>
            <button className="download-btn" onClick={downloadJson}>
              Download JSON
            </button>
          </section>
        )}
      </aside>

      <footer className="footer">
        <PushToTalkButton
          status={status}
          onStart={startRecording}
          onStop={stopRecording}
        />
        <p className="ptt-hint">
          Hold the button to record, release to send. The bot replies in
          the language you spoke (English or Hungarian).
        </p>
      </footer>
    </div>
  );
}

function ChatBubble({ turn }: { turn: TranscriptTurn }) {
  return (
    <div className={`bubble ${turn.role}`}>
      {turn.text}
      <div className="meta">
        <span className="tag">{turn.language}</span>
        {turn.sentiment && (
          <span className={`tag ${turn.sentiment.label}`}>
            {turn.sentiment.label} ({(turn.sentiment.score * 100).toFixed(0)}%)
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
  const disabled =
    status !== "ready" && status !== "recording";
  const recording = status === "recording";

  const handlePointerDown = (event: React.PointerEvent<HTMLButtonElement>) => {
    event.preventDefault();
    if (status === "ready") onStart();
  };
  const handlePointerUp = (event: React.PointerEvent<HTMLButtonElement>) => {
    event.preventDefault();
    if (status === "recording") onStop();
  };

  return (
    <button
      className={`ptt-button ${recording ? "recording" : ""}`}
      onPointerDown={handlePointerDown}
      onPointerUp={handlePointerUp}
      onPointerLeave={(e) => status === "recording" && handlePointerUp(e)}
      disabled={disabled}
    >
      {recording ? "Release" : "Hold"}
    </button>
  );
}
