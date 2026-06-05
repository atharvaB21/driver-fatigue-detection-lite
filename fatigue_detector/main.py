"""Driver Fatigue & Expression Detection System — Main Entry Point.

Real-time driver monitoring system that detects drowsiness, yawning,
and distraction using webcam feed with MediaPipe facial landmarks and
OpenCV computer vision.

Includes a --low-power mode optimized for weak hardware (Intel NUC,
Raspberry Pi, etc.) that reduces processing resolution, skips frames,
and disables expensive visual overlays to achieve usable FPS.

Usage:
    python main.py                      # Normal mode (default webcam)
    python main.py --low-power          # Low-power mode for NUC/SBC
    python main.py --source 1           # Use webcam at index 1
    python main.py --source video.mp4   # Use video file
    python main.py --process-width 240  # Custom processing resolution
    python main.py --frame-skip 3       # Process every 3rd frame

Controls:
    q     - Quit
    r     - Reset state machine
    l     - Toggle landmark display
    h     - Toggle HUD display
    s     - Take screenshot

Author: Driver Fatigue Detection System
"""

import sys
import os
import time
import argparse

import cv2
import numpy as np

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from detector.face import FaceDetector
from detector.ear import compute_ear
from detector.mar import mouth_aspect_ratio
from detector.head_pose import estimate_head_pose, get_camera_matrix, draw_pose_axes
from detector.perclos import PERCLOS
from detector.decision import DecisionEngine, DriverState
from alerts.alerter import Alerter
from utils.logger import FatigueLogger


def check_prerequisites() -> bool:
    """Check that required files exist before starting."""
    # Generate sound files if missing
    beep_path = os.path.join(Config.SOUNDS_DIR, 'beep.wav')
    if not os.path.exists(beep_path):
        print("[INFO] Sound files not found. Generating...")
        try:
            from generate_sounds import generate_all
            generate_all()
            print("[INFO] Sound files generated successfully.")
        except Exception as e:
            print(f"[WARN] Could not generate sounds: {e}")
            print("[WARN] Audio alerts will be disabled.")

    return True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Driver Fatigue & Expression Detection System"
    )
    parser.add_argument(
        '--source', default=None,
        help='Camera source: webcam index (0,1,...) or video file path. '
             'Default: uses config.py CAMERA_SOURCE'
    )
    parser.add_argument(
        '--no-audio', action='store_true',
        help='Disable audio alerts'
    )
    parser.add_argument(
        '--expression', action='store_true',
        help='Enable CNN expression detection (requires DeepFace)'
    )
    parser.add_argument(
        '--width', type=int, default=None,
        help='Display frame width (default: 640)'
    )

    # ── Low-power / NUC optimization flags ──
    parser.add_argument(
        '--low-power', action='store_true',
        help='Enable low-power mode for weak hardware (NUC, Raspberry Pi). '
             'Reduces processing resolution, skips frames, and disables '
             'expensive overlays for higher FPS.'
    )
    parser.add_argument(
        '--process-width', type=int, default=None,
        help='Processing frame width for MediaPipe inference. '
             'Lower = faster. Only used with --low-power or standalone. '
             'Default: 320 in low-power mode, same as display otherwise.'
    )
    parser.add_argument(
        '--frame-skip', type=int, default=None,
        help='Process every Nth frame with MediaPipe, reuse cached '
             'landmarks for skipped frames. Default: 2 in low-power mode.'
    )
    parser.add_argument(
        '--display-width', type=int, default=None,
        help='Display output width. In low-power mode this can be set '
             'independently of processing width. Default: 480 in '
             'low-power mode, 640 otherwise.'
    )

    return parser.parse_args()


def main():
    """Main application loop."""
    args = parse_args()

    # ── Determine effective settings ──
    low_power = args.low_power or Config.LOW_POWER_MODE

    print("=" * 60)
    print("  Driver Fatigue & Expression Detection System")
    if low_power:
        print("  ⚡ LOW-POWER MODE ACTIVE")
    print("=" * 60)
    print()

    # ── Check prerequisites ──
    if not check_prerequisites():
        sys.exit(1)

    # ── Configuration ──
    camera_source = Config.CAMERA_SOURCE
    if args.source is not None:
        # Try to parse as int (webcam index) or use as string (file path)
        try:
            camera_source = int(args.source)
        except ValueError:
            camera_source = args.source

    # Resolve display and processing widths
    if low_power:
        display_width = args.display_width or args.width or 480
        process_width = args.process_width or Config.PROCESS_WIDTH  # 320
        frame_skip = args.frame_skip or Config.FRAME_SKIP           # 2
        det_confidence = Config.LOW_POWER_DETECTION_CONFIDENCE      # 0.35
        trk_confidence = Config.LOW_POWER_TRACKING_CONFIDENCE       # 0.35
        show_landmarks = Config.LOW_POWER_SHOW_LANDMARKS            # False
        show_hud = Config.LOW_POWER_SHOW_HUD                        # True
    else:
        display_width = args.width or Config.FRAME_WIDTH  # 640
        process_width = args.process_width or display_width
        frame_skip = args.frame_skip or 1  # No skipping
        det_confidence = 0.5
        trk_confidence = 0.5
        show_landmarks = Config.SHOW_LANDMARKS
        show_hud = Config.SHOW_HUD

    # ── Initialize camera ──
    print(f"[INFO] Opening camera source: {camera_source}")
    cap = cv2.VideoCapture(camera_source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera source: {camera_source}")
        sys.exit(1)

    # Set camera properties — request display resolution from camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(display_width * 3 / 4))

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera opened: {actual_w}x{actual_h}")

    # ── Print optimization settings ──
    if low_power:
        print(f"[INFO] Display resolution:    {display_width}px wide")
        print(f"[INFO] Processing resolution: {process_width}px wide")
        print(f"[INFO] Frame skip:            process every {frame_skip} frame(s)")
        print(f"[INFO] Detection confidence:  {det_confidence}")
        print(f"[INFO] Landmarks drawing:     {'ON' if show_landmarks else 'OFF'}")
    else:
        print(f"[INFO] Resolution: {display_width}px wide")

    # ── Initialize face detector (MediaPipe — no model download needed) ──
    print("[INFO] Initializing MediaPipe Face Mesh...")
    face_detector = FaceDetector(
        min_detection_confidence=det_confidence,
        min_tracking_confidence=trk_confidence,
    )
    print("[INFO] Face detector ready.")

    # ── Initialize modules ──
    perclos_tracker = PERCLOS(window_size=Config.PERCLOS_WINDOW)

    decision_engine = DecisionEngine(
        ear_threshold=Config.EAR_THRESHOLD,
        drowsy_frames=Config.EAR_CONSEC_FRAMES_DROWSY,
        very_drowsy_frames=Config.EAR_CONSEC_FRAMES_VERY_DROWSY,
        asleep_frames=Config.EAR_CONSEC_FRAMES_ASLEEP,
        mar_threshold=Config.MAR_THRESHOLD,
        mar_consec_frames=Config.MAR_CONSEC_FRAMES,
        pitch_threshold=Config.HEAD_PITCH_THRESHOLD,
        yaw_threshold=Config.HEAD_YAW_THRESHOLD,
        perclos_threshold=Config.PERCLOS_THRESHOLD,
    )

    alerter = Alerter(
        sounds_dir=Config.SOUNDS_DIR,
        tts_cooldown=Config.TTS_COOLDOWN,
    )

    logger = FatigueLogger(db_path=Config.DB_PATH)

    # ── Optional CNN expression detector ──
    expression_detector = None
    if not low_power and (args.expression or Config.EXPRESSION_ENABLED):
        from expression.cnn_expression import ExpressionDetector
        expression_detector = ExpressionDetector(
            interval=Config.EXPRESSION_INTERVAL
        )
        if expression_detector.available:
            print("[INFO] CNN expression detection enabled.")
        else:
            print("[WARN] DeepFace not available. Expression detection disabled.")
            expression_detector = None
    elif low_power and (args.expression or Config.EXPRESSION_ENABLED):
        print("[INFO] CNN expression detection disabled in low-power mode.")

    # ── FPS tracking ──
    fps = 0.0
    frame_count = 0
    fps_start_time = time.time()
    no_face_start = None

    # ── Frame skip state ──
    skip_counter = 0
    cached_landmarks_list = []   # Cached landmarks from last processed frame
    cached_ear = 0.3
    cached_mar = 0.0
    cached_pitch = 0.0
    cached_yaw = 0.0
    cached_roll = 0.0
    cached_rvec = None
    cached_tvec = None

    print()
    print("[INFO] System running. Press 'q' to quit.")
    print("[INFO] Controls: r=reset, l=landmarks, h=HUD, s=screenshot")
    print()

    # ══════════════════════ MAIN LOOP ══════════════════════
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to read frame. End of video or camera error.")
                break

            # Resize frame to display width
            h_orig, w_orig = frame.shape[:2]
            if w_orig != display_width:
                aspect = h_orig / w_orig
                display_h = int(display_width * aspect)
                frame = cv2.resize(frame, (display_width, display_h))
            else:
                display_h = h_orig

            # ── Determine if we should run MediaPipe this frame ──
            skip_counter += 1
            should_process = (skip_counter >= frame_skip)
            if should_process:
                skip_counter = 0

            # ── Face detection + landmark extraction ──
            if should_process:
                if process_width < display_width:
                    # Downscale for faster inference, scale landmarks
                    # back to display coordinates
                    p_aspect = display_h / display_width
                    p_h = int(process_width * p_aspect)
                    process_frame = cv2.resize(
                        frame, (process_width, p_h),
                        interpolation=cv2.INTER_LINEAR
                    )
                    face_landmarks_list = face_detector.detect_and_get_landmarks(
                        process_frame,
                        scale_to=(display_width, display_h)
                    )
                else:
                    # Normal: process at display resolution
                    face_landmarks_list = face_detector.detect_and_get_landmarks(
                        frame
                    )
                cached_landmarks_list = face_landmarks_list
            else:
                # Reuse cached landmarks from last processed frame
                face_landmarks_list = cached_landmarks_list

            # Default values if no face detected
            ear = 0.3
            mar = 0.0
            pitch, yaw, roll = 0.0, 0.0, 0.0
            perclos_val = perclos_tracker.value
            expression = ""
            rvec, tvec = None, None

            if len(face_landmarks_list) > 0:
                no_face_start = None

                # Use the first detected face
                landmarks = face_landmarks_list[0]

                # Draw face rectangle from landmarks
                face_detector.draw_face_rect(frame, landmarks, color=(0, 255, 0))

                # Draw landmarks if enabled
                if show_landmarks:
                    face_detector.draw_landmarks(frame, landmarks)

                # ── EAR calculation ──
                left_eye = face_detector.get_left_eye(landmarks)
                right_eye = face_detector.get_right_eye(landmarks)
                ear = compute_ear(left_eye, right_eye)

                # Draw eye contours
                left_hull = cv2.convexHull(left_eye)
                right_hull = cv2.convexHull(right_eye)
                cv2.drawContours(frame, [left_hull], -1, (0, 255, 255), 1)
                cv2.drawContours(frame, [right_hull], -1, (0, 255, 255), 1)

                # ── MAR calculation ──
                mouth_inner = face_detector.get_mouth_inner(landmarks)
                mar = mouth_aspect_ratio(mouth_inner)

                # Draw mouth contour
                mouth_hull = cv2.convexHull(mouth_inner)
                cv2.drawContours(frame, [mouth_hull], -1, (0, 255, 255), 1)

                # ── Head pose estimation ──
                pose_points = face_detector.get_head_pose_points(landmarks)
                (pitch, yaw, roll), rvec, tvec = estimate_head_pose(
                    pose_points, frame.shape
                )

                # Cache computed values for skipped frames
                if should_process:
                    cached_ear = ear
                    cached_mar = mar
                    cached_pitch = pitch
                    cached_yaw = yaw
                    cached_roll = roll
                    cached_rvec = rvec
                    cached_tvec = tvec

                # Draw pose axes if we got valid results
                if rvec is not None and not low_power:
                    cam_matrix, dist_coeffs = get_camera_matrix(frame.shape)
                    draw_pose_axes(frame, rvec, tvec, cam_matrix, dist_coeffs)

                # ── PERCLOS update ──
                perclos_val = perclos_tracker.update(ear, Config.EAR_THRESHOLD)

                # ── Decision engine ──
                state, distracted, yawning = decision_engine.update(
                    ear, mar, pitch, yaw, perclos_val
                )

                # ── Optional CNN expression ──
                if expression_detector is not None:
                    expression_detector.update(frame)
                    expression = expression_detector.get_emotion()

                # ── Logging ──
                logger.log_state_change(
                    state, ear, mar, perclos_val, pitch, yaw,
                    distracted, yawning, expression
                )
                logger.log_snapshot(
                    state, ear, mar, perclos_val, pitch, yaw,
                    distracted, yawning, expression,
                    interval=Config.LOG_SNAPSHOT_INTERVAL
                )

                # ── Alerts ──
                if not args.no_audio:
                    alerter.play_alert(state, distracted)

                # ── Visual overlays ──
                if show_hud:
                    alerter.draw_overlay(
                        frame, state, distracted, yawning,
                        ear, mar, perclos_val, pitch, yaw, fps
                    )

                # ── Expression label ──
                if expression and expression != "N/A":
                    cv2.putText(
                        frame, f"Expr: {expression}",
                        (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 200, 0), 2
                    )

            else:
                # No face detected
                if no_face_start is None:
                    no_face_start = time.time()

                elapsed = time.time() - no_face_start

                if elapsed > 2.0:
                    # Show warning after 2 seconds of no face
                    cv2.putText(
                        frame, "NO FACE DETECTED",
                        (frame.shape[1] // 2 - 130, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 0, 255), 2
                    )
                    cv2.putText(
                        frame, "Please face the camera",
                        (frame.shape[1] // 2 - 140, frame.shape[0] // 2 + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 140, 255), 2
                    )

            # ── FPS calculation ──
            frame_count += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start_time = time.time()

            # ── Display ──
            cv2.imshow("Driver Fatigue Monitor", frame)

            # ── Key handling ──
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("[INFO] Quit requested.")
                break
            elif key == ord('r'):
                decision_engine.reset()
                perclos_tracker.reset()
                print("[INFO] State machine reset.")
            elif key == ord('l'):
                show_landmarks = not show_landmarks
                print(f"[INFO] Landmarks {'ON' if show_landmarks else 'OFF'}")
            elif key == ord('h'):
                show_hud = not show_hud
                print(f"[INFO] HUD {'ON' if show_hud else 'OFF'}")
            elif key == ord('s'):
                screenshot_path = os.path.join(
                    Config.BASE_DIR,
                    f"screenshot_{int(time.time())}.png"
                )
                cv2.imwrite(screenshot_path, frame)
                print(f"[INFO] Screenshot saved: {screenshot_path}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        # ── Cleanup ──
        print("[INFO] Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        alerter.cleanup()
        face_detector.close()

        # Print session summary
        summary = logger.get_session_summary()
        if summary and summary[0] > 0:
            print()
            print("=" * 60)
            print("  SESSION SUMMARY")
            print("=" * 60)
            print(f"  Total events logged:  {summary[0]}")
            print(f"  Drowsy alerts:        {summary[1]}")
            print(f"  Very drowsy alerts:   {summary[2]}")
            print(f"  Asleep alerts:        {summary[3]}")
            print(f"  Distraction events:   {summary[4]}")
            print(f"  Yawn events:          {summary[5]}")
            print(f"  Average EAR:          {summary[6]:.3f}" if summary[6] else "")
            print(f"  Average PERCLOS:      {summary[7]:.3f}" if summary[7] else "")
            print(f"  Session start:        {summary[8]}")
            print(f"  Session end:          {summary[9]}")
            print("=" * 60)
            print(f"  Data saved to: {Config.DB_PATH}")
            print(f"  View dashboard: streamlit run dashboard/app.py")
            print("=" * 60)

        print("[INFO] Done.")


if __name__ == '__main__':
    main()
