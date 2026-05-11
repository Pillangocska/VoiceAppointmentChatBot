"""Microphone capture utilities for hold-to-record interaction.

Provides a blocking recorder that streams audio from the default input
device while the caller is holding the record key, and returns the
captured signal as a single NumPy array suitable for ASR.
"""

import queue
import sys
from typing import List

import numpy as np
import sounddevice as sd

from voiceappointmentchatbot.config import AudioConfig


class HoldToRecord:
    """Blocking microphone recorder driven by the Enter key.

    The user presses Enter to start capture and presses Enter again to
    stop. Captured frames are concatenated into a single mono float32
    array at the configured sample rate.

    Attributes:
        config: Audio capture configuration.
    """

    def __init__(self, config: AudioConfig) -> None:
        """Initialise the recorder with the given audio configuration."""
        self.config = config
        self._frames: queue.Queue[np.ndarray] = queue.Queue()

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """Push a copy of each input block onto the frame queue."""
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        self._frames.put(indata.copy())

    def record(self) -> np.ndarray:
        """Record audio between two Enter key presses.

        Returns:
            One-dimensional float32 array of PCM samples at the
            configured sample rate.
        """
        input("Press Enter to start recording...")
        print("Recording... press Enter again to stop.")

        with sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype=self.config.dtype,
            blocksize=self.config.block_size,
            callback=self._callback,
        ):
            input()

        print("Stopped.")
        chunks: List[np.ndarray] = []
        while not self._frames.empty():
            chunks.append(self._frames.get())

        if not chunks:
            return np.zeros(0, dtype=np.float32)

        audio = np.concatenate(chunks, axis=0)
        return audio.flatten().astype(np.float32)
