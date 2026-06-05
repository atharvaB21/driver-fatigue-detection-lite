"""Escalating alert system with visual overlays and threaded audio.

Provides visual feedback (colored overlays, banners, HUD) and audio
alerts (beep sounds, alarm, text-to-speech) that escalate based on
the driver's current state.
"""

import cv2
import numpy as np
import threading
import time
import os

# Optional audio dependencies — degrade gracefully if not installed
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

try:
    import pygame
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except (ImportError, Exception):
    PYGAME_AVAILABLE = False

from detector.decision import DriverState


class Alerter:
    """Escalating alert manager with visual and audio feedback.

    Visual alerts are drawn directly on OpenCV frames.
    Audio alerts run in daemon threads to avoid blocking the video loop.

    Parameters
    ----------
    sounds_dir : str
        Path to directory containing beep.wav and alarm.wav.
    tts_cooldown : float
        Minimum seconds between TTS alerts to prevent spam.
    """

    # Color constants (BGR format for OpenCV)
    COLOR_GREEN   = (0, 200, 0)
    COLOR_AMBER   = (0, 191, 255)
    COLOR_RED     = (0, 0, 255)
    COLOR_ORANGE  = (0, 140, 255)
    COLOR_YELLOW  = (0, 255, 255)
    COLOR_WHITE   = (255, 255, 255)
    COLOR_BLACK   = (0, 0, 0)
    COLOR_DARK_BG = (40, 40, 40)

    def __init__(self, sounds_dir: str, tts_cooldown: float = 5.0):
        self.sounds_dir = sounds_dir
        self.tts_cooldown = tts_cooldown
        self.last_tts_time = 0.0
        self.last_beep_time = 0.0

        # Initialize TTS engine
        self.tts_engine = None
        if TTS_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 180)
                self.tts_engine.setProperty('volume', 1.0)
            except Exception:
                self.tts_engine = None

    # ──────────────────────── Visual Overlays ────────────────────────

    def draw_overlay(self, frame: np.ndarray, state: DriverState,
                     distracted: bool, yawning: bool,
                     ear: float, mar: float, perclos: float,
                     pitch: float, yaw: float, fps: float) -> np.ndarray:
        """Draw all visual alerts and HUD on the video frame.

        Parameters
        ----------
        frame : np.ndarray
            The video frame to draw on (modified in-place and returned).
        state : DriverState
            Current driver state from the decision engine.
        distracted : bool
            Whether the driver is currently distracted.
        yawning : bool
            Whether the driver is currently yawning.
        ear, mar, perclos, pitch, yaw, fps : float
            Current metric values for HUD display.

        Returns
        -------
        np.ndarray
            The frame with overlays drawn.
        """
        h, w = frame.shape[:2]

        # ── State-specific overlays ──
        if state == DriverState.ASLEEP:
            # Flashing red border
            flash = int(time.time() * 4) % 2 == 0
            border_color = self.COLOR_RED if flash else self.COLOR_BLACK
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_color, 8)
            self._draw_alert_banner(frame, "!!! WAKE UP !!!", self.COLOR_RED, large=True)

        elif state == DriverState.VERY_DROWSY:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), self.COLOR_RED, -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), self.COLOR_RED, 4)
            self._draw_alert_banner(frame, "VERY DROWSY - PULL OVER!", self.COLOR_RED)

        elif state == DriverState.DROWSY:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), self.COLOR_AMBER, -1)
            cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)
            self._draw_alert_banner(frame, "DROWSY - STAY ALERT", self.COLOR_AMBER)

        else:
            self._draw_status_badge(frame, "ALERT", self.COLOR_GREEN)

        # ── Distraction overlay ──
        if distracted:
            self._draw_alert_banner(frame, "EYES ON THE ROAD!", self.COLOR_ORANGE, y_offset=80)

        # ── Yawning label ──
        if yawning:
            self._draw_alert_banner(frame, "YAWNING DETECTED", self.COLOR_YELLOW, y_offset=120)

        # ── HUD Panel ──
        self._draw_hud(frame, ear, mar, perclos, pitch, yaw, fps, state)

        return frame

    def _draw_alert_banner(self, frame: np.ndarray, text: str, color: tuple,
                           y_offset: int = 30, large: bool = False) -> None:
        """Draw a centered alert banner with background."""
        h, w = frame.shape[:2]
        font_scale = 1.2 if large else 0.8
        thickness = 3 if large else 2
        font = cv2.FONT_HERSHEY_SIMPLEX

        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        x = (w - text_size[0]) // 2
        y = y_offset + text_size[1]
        pad = 10

        # Dark background with colored border
        cv2.rectangle(frame,
                      (x - pad, y - text_size[1] - pad),
                      (x + text_size[0] + pad, y + pad),
                      self.COLOR_DARK_BG, -1)
        cv2.rectangle(frame,
                      (x - pad, y - text_size[1] - pad),
                      (x + text_size[0] + pad, y + pad),
                      color, 2)
        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)

    def _draw_status_badge(self, frame: np.ndarray, text: str, color: tuple) -> None:
        """Draw a small status badge in the top-left corner."""
        cv2.rectangle(frame, (10, 10), (130, 45), self.COLOR_DARK_BG, -1)
        cv2.rectangle(frame, (10, 10), (130, 45), color, 2)
        cv2.putText(frame, text, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def _draw_hud(self, frame: np.ndarray, ear: float, mar: float,
                  perclos: float, pitch: float, yaw: float,
                  fps: float, state: DriverState) -> None:
        """Draw a semi-transparent HUD panel with real-time metrics."""
        h, w = frame.shape[:2]
        panel_w, panel_h = 220, 200
        x0 = w - panel_w - 10
        y0 = 10

        # Semi-transparent dark background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h),
                      self.COLOR_DARK_BG, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h),
                      (100, 100, 100), 1)

        # Metric rows with color-coded values
        metrics = [
            (f"FPS: {fps:.0f}",         self.COLOR_WHITE),
            (f"EAR: {ear:.3f}",          self.COLOR_GREEN if ear > 0.25 else self.COLOR_RED),
            (f"MAR: {mar:.3f}",          self.COLOR_GREEN if mar < 0.60 else self.COLOR_YELLOW),
            (f"PERCLOS: {perclos:.2f}",  self.COLOR_GREEN if perclos < 0.15 else self.COLOR_RED),
            (f"Pitch: {pitch:.1f}",      self.COLOR_GREEN if abs(pitch) < 20 else self.COLOR_ORANGE),
            (f"Yaw: {yaw:.1f}",          self.COLOR_GREEN if abs(yaw) < 30 else self.COLOR_ORANGE),
            (f"State: {state.name}",     self._state_color(state)),
        ]

        for i, (text, color) in enumerate(metrics):
            cv2.putText(frame, text, (x0 + 10, y0 + 25 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def _state_color(self, state: DriverState) -> tuple:
        """Get the display color for a given driver state."""
        return {
            DriverState.ALERT:       self.COLOR_GREEN,
            DriverState.DROWSY:      self.COLOR_AMBER,
            DriverState.VERY_DROWSY: self.COLOR_RED,
            DriverState.ASLEEP:      self.COLOR_RED,
        }.get(state, self.COLOR_WHITE)

    # ──────────────────────── Audio Alerts ────────────────────────

    def play_alert(self, state: DriverState, distracted: bool) -> None:
        """Play appropriate audio alert in background thread.

        Audio is rate-limited to prevent overlapping sounds.

        Parameters
        ----------
        state : DriverState
            Current driver state.
        distracted : bool
            Whether the driver is distracted.
        """
        now = time.time()

        if state == DriverState.ASLEEP:
            if now - self.last_tts_time > self.tts_cooldown:
                self._speak_async("Wake up! Pull over immediately!")
                self.last_tts_time = now
            self._play_sound_async(os.path.join(self.sounds_dir, 'alarm.wav'))

        elif state == DriverState.VERY_DROWSY:
            if now - self.last_beep_time > 2.0:
                self._play_sound_async(os.path.join(self.sounds_dir, 'beep.wav'))
                self.last_beep_time = now

        if distracted:
            if now - self.last_tts_time > self.tts_cooldown:
                self._speak_async("Eyes on the road!")
                self.last_tts_time = now
            if now - self.last_beep_time > 2.0:
                self._play_sound_async(os.path.join(self.sounds_dir, 'beep.wav'))
                self.last_beep_time = now

    def _play_sound_async(self, path: str) -> None:
        """Play a WAV file asynchronously using pygame."""
        if not PYGAME_AVAILABLE or not os.path.exists(path):
            return

        def _play():
            try:
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
            except Exception:
                pass

        threading.Thread(target=_play, daemon=True).start()

    def _speak_async(self, text: str) -> None:
        """Speak text asynchronously using pyttsx3 TTS."""
        if self.tts_engine is None:
            return

        def _speak():
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception:
                pass

        threading.Thread(target=_speak, daemon=True).start()

    # ──────────────────────── Cleanup ────────────────────────

    def cleanup(self) -> None:
        """Clean up audio resources."""
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.quit()
            except Exception:
                pass
