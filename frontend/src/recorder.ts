// Browser push-to-talk audio capture wrapping MediaRecorder.
//
// The bot's server-side pipeline accepts anything ffmpeg can decode, so
// we let the browser pick its native MediaRecorder mime type (Chrome
// emits audio/webm;codecs=opus, Safari WAV) and forward the blob as-is.

export class PushToTalkRecorder {
  private stream: MediaStream | null = null;
  private recorder: MediaRecorder | null = null;
  private chunks: Blob[] = [];

  async start(): Promise<void> {
    if (this.recorder) return;
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this.chunks = [];
    this.recorder = new MediaRecorder(this.stream);
    this.recorder.ondataavailable = (event) => {
      if (event.data.size > 0) this.chunks.push(event.data);
    };
    this.recorder.start();
  }

  async stop(): Promise<Blob | null> {
    const recorder = this.recorder;
    if (!recorder) return null;
    const stopped = new Promise<void>((resolve) => {
      recorder.onstop = () => resolve();
    });
    recorder.stop();
    await stopped;
    this.stream?.getTracks().forEach((track) => track.stop());
    const mime = recorder.mimeType || "audio/webm";
    const blob = new Blob(this.chunks, { type: mime });
    this.recorder = null;
    this.stream = null;
    this.chunks = [];
    return blob.size > 0 ? blob : null;
  }

  cancel(): void {
    try {
      this.recorder?.stop();
    } catch {
      // ignore
    }
    this.stream?.getTracks().forEach((track) => track.stop());
    this.recorder = null;
    this.stream = null;
    this.chunks = [];
  }
}

export async function blobToBase64(blob: Blob): Promise<string> {
  const buf = await blob.arrayBuffer();
  const bytes = new Uint8Array(buf);
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

export function base64ToWavUrl(b64: string): string {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  const blob = new Blob([bytes], { type: "audio/wav" });
  return URL.createObjectURL(blob);
}
