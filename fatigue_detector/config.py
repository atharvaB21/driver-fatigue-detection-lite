"""
Central configuration for the Driver Fatigue Detection System.

All tunable parameters — thresholds, paths, display options — are
consolidated here so every other module can simply ``from config import Config``.
"""

import os


class Config:
    """Application-wide configuration constants.

    Attributes are grouped by subsystem.  Modify values here (or override
    at runtime) to tune detection sensitivity, camera source, file paths,
    alert behaviour, and display preferences.
    """

    # ------------------------------------------------------------------ #
    #  EAR  (Eye Aspect Ratio)                                           #
    # ------------------------------------------------------------------ #
    EAR_THRESHOLD: float = 0.25
    """Eyes are considered *closed* when EAR drops below this value."""

    EAR_CONSEC_FRAMES_DROWSY: int = 20
    """Consecutive closed-eye frames to trigger a **drowsy** warning."""

    EAR_CONSEC_FRAMES_VERY_DROWSY: int = 50
    """Consecutive closed-eye frames to trigger a **very drowsy** warning."""

    EAR_CONSEC_FRAMES_ASLEEP: int = 100
    """Consecutive closed-eye frames to trigger an **asleep** alarm."""

    # ------------------------------------------------------------------ #
    #  MAR  (Mouth Aspect Ratio)                                         #
    # ------------------------------------------------------------------ #
    MAR_THRESHOLD: float = 0.60
    """Mouth is considered *open/yawning* when MAR exceeds this value."""

    MAR_CONSEC_FRAMES: int = 15
    """Consecutive yawn frames before a yawn event is registered."""

    # ------------------------------------------------------------------ #
    #  PERCLOS  (Percentage of Eye Closure)                               #
    # ------------------------------------------------------------------ #
    PERCLOS_WINDOW: int = 60
    """Sliding window size (in seconds) over which PERCLOS is computed."""

    PERCLOS_THRESHOLD: float = 0.15
    """PERCLOS ratio above which fatigue is flagged."""

    # ------------------------------------------------------------------ #
    #  Head Pose                                                         #
    # ------------------------------------------------------------------ #
    HEAD_PITCH_THRESHOLD: float = 20.0
    """Max allowable pitch (degrees) before a head-nod alert fires."""

    HEAD_YAW_THRESHOLD: float = 30.0
    """Max allowable yaw (degrees) before a head-turn alert fires."""

    # ------------------------------------------------------------------ #
    #  Camera                                                            #
    # ------------------------------------------------------------------ #
    CAMERA_SOURCE: int = 0
    """OpenCV VideoCapture index (0 = default webcam)."""

    FRAME_WIDTH: int = 640
    """Desired frame width in pixels (height scales proportionally)."""

    # ------------------------------------------------------------------ #
    #  Low-Power / NUC Performance Optimization                          #
    # ------------------------------------------------------------------ #
    LOW_POWER_MODE: bool = False
    """Master switch for low-power optimizations (e.g. for NUC/SBC).
    When True, applies all low-power overrides below automatically."""

    PROCESS_WIDTH: int = 320
    """Frame width used for MediaPipe inference in low-power mode.
    Lower = faster inference. The display frame stays at FRAME_WIDTH."""

    FRAME_SKIP: int = 2
    """In low-power mode, only run MediaPipe every Nth frame.
    Intermediate frames reuse the last detected landmarks.
    1 = process every frame (no skip), 2 = every other frame, etc."""

    LOW_POWER_DETECTION_CONFIDENCE: float = 0.35
    """Lower detection confidence for faster inference in low-power mode."""

    LOW_POWER_TRACKING_CONFIDENCE: float = 0.35
    """Lower tracking confidence for faster inference in low-power mode."""

    LOW_POWER_SHOW_LANDMARKS: bool = False
    """Disable landmark drawing in low-power mode to save CPU."""

    LOW_POWER_SHOW_HUD: bool = True
    """Keep HUD on in low-power mode (lightweight operation)."""

    # ------------------------------------------------------------------ #
    #  Paths                                                             #
    # ------------------------------------------------------------------ #
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    """Root directory of the fatigue_detector package."""

    MODEL_PATH: str = os.path.join(
        BASE_DIR, "models", "face_landmarker.task"
    )
    """Path to the MediaPipe face landmarker model."""

    DB_PATH: str = os.path.join(BASE_DIR, "fatigue_log.db")
    """SQLite database used for persisting fatigue-event logs."""

    SOUNDS_DIR: str = os.path.join(BASE_DIR, "sounds")
    """Directory containing alert audio files."""

    # ------------------------------------------------------------------ #
    #  Alerts                                                            #
    # ------------------------------------------------------------------ #
    TTS_COOLDOWN: float = 5.0
    """Minimum interval (seconds) between consecutive TTS alerts."""

    ALERT_BEEP_PATH: str = os.path.join(SOUNDS_DIR, "beep.wav")
    """Path to the short beep sound effect."""

    ALERT_ALARM_PATH: str = os.path.join(SOUNDS_DIR, "alarm.wav")
    """Path to the sustained alarm sound effect."""

    # ------------------------------------------------------------------ #
    #  CNN Expression Recognition  (optional)                            #
    # ------------------------------------------------------------------ #
    EXPRESSION_ENABLED: bool = False
    """Enable / disable the CNN-based expression classifier."""

    EXPRESSION_INTERVAL: int = 5
    """Run expression inference every *N* frames (performance knob)."""

    # ------------------------------------------------------------------ #
    #  Logging                                                           #
    # ------------------------------------------------------------------ #
    LOG_SNAPSHOT_INTERVAL: int = 30
    """Seconds between periodic metric snapshots written to the DB."""

    # ------------------------------------------------------------------ #
    #  Display / HUD                                                     #
    # ------------------------------------------------------------------ #
    SHOW_LANDMARKS: bool = True
    """Draw facial landmarks on the video feed."""

    SHOW_HUD: bool = True
    """Overlay the heads-up display (EAR, MAR, PERCLOS gauges)."""
