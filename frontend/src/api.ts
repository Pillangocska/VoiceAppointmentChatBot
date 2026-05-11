import type { ServerEvent } from "./types";

export type Connection = {
  socket: WebSocket;
  send: (msg: object) => void;
  close: () => void;
};

export function connectSession(
  onEvent: (event: ServerEvent) => void,
  onError: (err: Event | Error) => void,
  onClose: () => void,
): Connection {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  const url = `${proto}://${window.location.host}/ws`;
  const socket = new WebSocket(url);

  socket.onmessage = (event) => {
    try {
      const parsed = JSON.parse(event.data) as ServerEvent;
      onEvent(parsed);
    } catch (err) {
      onError(err as Error);
    }
  };
  socket.onerror = (event) => onError(event);
  socket.onclose = () => onClose();

  return {
    socket,
    send: (msg) => {
      if (socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify(msg));
      }
    },
    close: () => socket.close(),
  };
}
