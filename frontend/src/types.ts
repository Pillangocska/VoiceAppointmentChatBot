export type Lang = "en" | "hu";

export interface SlotSpec {
  name: string;
  type: string | null;
  prompt: Record<Lang, string>;
}

export interface DomainPayload {
  name: string;
  display_name: Record<Lang, string>;
  slots: SlotSpec[];
}

export interface BookingPayload {
  slots: Record<string, string>;
  confirmed: string[];
  pending_phone_confirmation: boolean;
  missing: string[];
  is_complete: boolean;
}

export interface SentimentPayload {
  label: string;
  score: number;
}

export interface TranscriptTurn {
  role: "user" | "assistant";
  text: string;
  language: string;
  sentiment?: SentimentPayload;
}

export type ServerEvent =
  | {
      type: "session_ready";
      domain: DomainPayload;
      booking: BookingPayload;
    }
  | {
      type: "transcript";
      text: string;
      language: string;
      language_probability: number;
    }
  | { type: "sentiment"; label: string; score: number }
  | {
      type: "reply";
      text: string;
      language: string;
      audio_base64: string | null;
      booking: BookingPayload;
      booking_complete: boolean;
    }
  | {
      type: "saved";
      path: string | null;
      payload: Record<string, unknown>;
    }
  | { type: "error"; message: string };
