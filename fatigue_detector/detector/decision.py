"""Decision engine with Finite State Machine for driver state classification.

Implements a 4-state FSM (ALERT → DROWSY → VERY_DROWSY → ASLEEP) with
orthogonal DISTRACTED and YAWNING flags. State transitions are based on
consecutive EAR frames below threshold, PERCLOS values, MAR for yawning,
and head pose angles for distraction detection.
"""

from enum import IntEnum
import time


class DriverState(IntEnum):
    """Enumeration of possible driver alertness states.

    States are ordered by severity so comparisons like
    ``state >= DriverState.DROWSY`` work intuitively.
    """
    ALERT = 0
    DROWSY = 1
    VERY_DROWSY = 2
    ASLEEP = 3


class DecisionEngine:
    """Finite State Machine for driver fatigue decision logic.

    Combines EAR (Eye Aspect Ratio), MAR (Mouth Aspect Ratio),
    PERCLOS, and head pose data to classify the driver's state and
    detect distraction/yawning.

    Parameters
    ----------
    ear_threshold : float
        EAR value below which eyes are considered closed.
    drowsy_frames : int
        Consecutive frames below EAR threshold to trigger DROWSY.
    very_drowsy_frames : int
        Consecutive frames below EAR threshold to trigger VERY_DROWSY.
    asleep_frames : int
        Consecutive frames below EAR threshold to trigger ASLEEP.
    mar_threshold : float
        MAR value above which the mouth is considered yawning.
    mar_consec_frames : int
        Consecutive frames above MAR threshold to confirm a yawn.
    pitch_threshold : float
        Head pitch angle (degrees) beyond which driver is distracted.
    yaw_threshold : float
        Head yaw angle (degrees) beyond which driver is distracted.
    perclos_threshold : float
        PERCLOS value above which fatigue is indicated.
    """

    def __init__(
        self,
        ear_threshold: float,
        drowsy_frames: int,
        very_drowsy_frames: int,
        asleep_frames: int,
        mar_threshold: float,
        mar_consec_frames: int,
        pitch_threshold: float,
        yaw_threshold: float,
        perclos_threshold: float
    ):
        self.ear_threshold = ear_threshold
        self.drowsy_frames = drowsy_frames
        self.very_drowsy_frames = very_drowsy_frames
        self.asleep_frames = asleep_frames
        self.mar_threshold = mar_threshold
        self.mar_consec_frames = mar_consec_frames
        self.pitch_threshold = pitch_threshold
        self.yaw_threshold = yaw_threshold
        self.perclos_threshold = perclos_threshold

        # Current state
        self.state = DriverState.ALERT
        self.ear_counter = 0
        self.yawn_counter = 0
        self.distracted = False
        self.yawning = False
        self.last_state_change = time.time()

    def update(self, ear: float, mar: float, pitch: float, yaw: float,
               perclos: float) -> tuple:
        """Update the state machine with new sensor readings.

        Parameters
        ----------
        ear : float
            Current average Eye Aspect Ratio.
        mar : float
            Current Mouth Aspect Ratio.
        pitch : float
            Current head pitch angle in degrees.
        yaw : float
            Current head yaw angle in degrees.
        perclos : float
            Current PERCLOS value (0.0 to 1.0).

        Returns
        -------
        tuple
            (state: DriverState, is_distracted: bool, is_yawning: bool)
        """
        # ── EAR-based state transitions ──
        if ear < self.ear_threshold:
            self.ear_counter += 1
        else:
            self.ear_counter = 0
            if self.state != DriverState.ALERT:
                self.state = DriverState.ALERT
                self.last_state_change = time.time()

        # Escalate state based on consecutive closed-eye frames
        if self.ear_counter >= self.asleep_frames:
            if self.state != DriverState.ASLEEP:
                self.state = DriverState.ASLEEP
                self.last_state_change = time.time()
        elif self.ear_counter >= self.very_drowsy_frames:
            if self.state != DriverState.VERY_DROWSY:
                self.state = DriverState.VERY_DROWSY
                self.last_state_change = time.time()
        elif self.ear_counter >= self.drowsy_frames:
            if self.state != DriverState.DROWSY:
                self.state = DriverState.DROWSY
                self.last_state_change = time.time()

        # ── PERCLOS can also trigger drowsiness ──
        if perclos > self.perclos_threshold and self.state == DriverState.ALERT:
            self.state = DriverState.DROWSY
            self.last_state_change = time.time()

        # ── MAR-based yawn detection ──
        if mar > self.mar_threshold:
            self.yawn_counter += 1
        else:
            if self.yawn_counter >= self.mar_consec_frames:
                self.yawning = True   # Confirmed yawn
            else:
                self.yawning = False
            self.yawn_counter = 0

        # Also flag yawning while mouth is still open
        if self.yawn_counter >= self.mar_consec_frames:
            self.yawning = True

        # ── Head pose distraction ──
        self.distracted = (
            abs(pitch) > self.pitch_threshold or
            abs(yaw) > self.yaw_threshold
        )

        return self.state, self.distracted, self.yawning

    def reset(self) -> None:
        """Reset the state machine to initial state."""
        self.state = DriverState.ALERT
        self.ear_counter = 0
        self.yawn_counter = 0
        self.distracted = False
        self.yawning = False
        self.last_state_change = time.time()

    @property
    def time_in_state(self) -> float:
        """Seconds elapsed since the last state transition."""
        return time.time() - self.last_state_change
