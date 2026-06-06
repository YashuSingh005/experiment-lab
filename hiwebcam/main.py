"""
Particle Galaxy - Hand Tracking Edition
Compatible: Python 3.8-3.13 | Windows/macOS/Linux
MediaPipe:  0.10.30+

pip install mediapipe opencv-python pygame numpy

Controls:
  Open hand   -> particles EXPLODE outward
  Closed fist -> particles IMPLODE / accumulate
  Move hand   -> swarm follows
  ESC         -> exit
"""

from __future__ import annotations
import os, sys, math, random, threading, traceback, logging, urllib.request, time
from typing import Optional, Tuple

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("swarm_log.txt", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("galaxy")

# Dependency check
log.info("Checking dependencies...")
_MISSING = []
try:
    import cv2
    log.info("  opencv  OK  (%s)", cv2.__version__)
except ImportError as e:
    log.error("  opencv MISSING: %s", e)
    _MISSING.append("opencv-python")
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    log.info("  mediapipe OK")
except ImportError as e:
    log.error("  mediapipe MISSING: %s", e)
    _MISSING.append("mediapipe")
try:
    import pygame
    log.info("  pygame  OK  (%s)", pygame.version.ver)
except ImportError as e:
    log.error("  pygame MISSING: %s", e)
    _MISSING.append("pygame")
try:
    import numpy as np
    log.info("  numpy   OK  (%s)", np.__version__)
except ImportError as e:
    log.error("  numpy MISSING: %s", e)
    _MISSING.append("numpy")

if _MISSING:
    log.error("Run:  pip install %s", " ".join(_MISSING))
    input("\nPress Enter to exit...")
    sys.exit(1)

# Model auto-download
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    log.info("Downloading hand model (~17 MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        log.info("Model saved: %s", MODEL_PATH)
    except Exception as e:
        log.error("Download failed: %s", e)
        input("Press Enter...")
        sys.exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================

NUM_PARTICLES   = 6000
FPS_TARGET      = 60
DAMPING_NORMAL  = 0.92
DAMPING_IMPLODE = 0.80
BASE_ATTRACT    = 1.8
IMPLODE_FORCE   = 4.5
EXPLODE_FORCE   = 6.0
LERP_SPEED      = 0.12
MAX_SPEED       = 22.0
TRAIL_ALPHA     = 18
OPEN_THRESHOLD  = 0.55

PALETTES = [
    ((20,  80, 255), (255,  40, 180)),
    ((0,  220, 120), (255, 200,   0)),
    ((180,  0, 255), (255,  80,   0)),
    ((0,  200, 255), (255, 255,  80)),
]

# =============================================================================
# PARTICLE
# =============================================================================

class Particle:
    __slots__ = ("theta", "phi", "r", "vx", "vy", "vz", "radius", "hue_offset")

    def __init__(self):
        self.theta = random.uniform(0, math.tau)
        self.phi   = random.uniform(0, math.pi)

        # sphere radius base
        self.r = random.uniform(180, 320)

        self.vx = self.vy = self.vz = 0.0
        self.radius = random.uniform(1.2, 2.8)
        self.hue_offset = random.random()

    def to_3d(self):
        x = self.r * math.sin(self.phi) * math.cos(self.theta)
        y = self.r * math.sin(self.phi) * math.sin(self.theta)
        z = self.r * math.cos(self.phi)
        return x, y, z

    def update(self, spin, pulse):
        # gentle rotation (always moving = trending feel)
        self.theta += spin
        self.phi   += spin * 0.6

        # pulse = breathing sphere
        self.r += math.sin(pulse + self.hue_offset * 5) * 0.3

# =============================================================================
# HAND TRACKER
# =============================================================================

class HandTracker:
    _PALM_IDS      = [0, 1, 5, 9, 13, 17]
    _FINGERTIP_IDS = [8, 12, 16, 20]

    def __init__(self, camera_index=0):
        self._state  = None
        self._lock   = threading.Lock()
        self._stop   = threading.Event()

        tracker_self = self

        def _on_result(result, _img, _ts):
            if not result.hand_landmarks:
                with tracker_self._lock:
                    tracker_self._state = None
                return
            lm = result.hand_landmarks[0]
            xs = [lm[i].x for i in tracker_self._PALM_IDS]
            ys = [lm[i].y for i in tracker_self._PALM_IDS]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            wrist = lm[0]
            tip_dists = [
                math.hypot(lm[i].x - wrist.x, lm[i].y - wrist.y)
                for i in tracker_self._FINGERTIP_IDS
            ]
            ref    = math.hypot(lm[9].x - wrist.x, lm[9].y - wrist.y) + 1e-6
            spread = (sum(tip_dists) / len(tip_dists)) / ref
            spread_norm = max(0.0, min(1.0, (spread - 1.0) / 0.8))
            is_open = spread_norm >= OPEN_THRESHOLD
            with tracker_self._lock:
                tracker_self._state = HandState(cx, cy, is_open, spread_norm)

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp_vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=_on_result,
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(options)
        log.info("HandLandmarker ready.")

        log.info("Opening webcam %d...", camera_index)
        self._cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            raise RuntimeError("Cannot open webcam %d" % camera_index)
        log.info("Webcam OK.")

        self._thread = threading.Thread(target=self._run, daemon=True, name="HandTracker")

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(3)
        self._cap.release()
        self._detector.close()

    def get_state(self):
        with self._lock:
            return self._state

    def _run(self):
        ts = 0
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if not ret:
                continue
            frame  = cv2.flip(frame, 1)
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts    += 1
            self._detector.detect_async(mp_img, ts)

# =============================================================================
# PARTICLE SWARM
# =============================================================================

class ParticleSwarm:
    def __init__(self, count, W, H):
        self.W, self.H = W, H
        self.particles = [Particle() for _ in range(count)]
        self.angle = 0.0
        self.pulse = 0.0

    def update(self, state, t_global):
        # always rotating (important for “trending 3D look”)
        self.angle += 0.003
        self.pulse  += 0.05

        # hand influence = distort sphere
        if state:
            strength = 1.0 if state.is_open else -1.0
            self.angle += strength * 0.01

            # faster breathing when open hand
            self.pulse += state.spread * 0.2

        for p in self.particles:
            p.update(self.angle, self.pulse)

    def draw(self, surface, palette, t_global):
        cx, cy = self.W // 2, self.H // 2

        for p in self.particles:
            x, y, z = p.to_3d()

            # fake camera depth
            perspective = 500 / (500 + z)

            sx = int(cx + x * perspective)
            sy = int(cy + y * perspective)

            # brightness based on depth
            brightness = max(0.3, min(1.2, perspective))

            c0, c1 = palette
            t = (math.sin(t_global + p.hue_offset * 6) + 1) * 0.5

            color = (
                int((c0[0] + t * (c1[0] - c0[0])) * brightness),
                int((c0[1] + t * (c1[1] - c0[1])) * brightness),
                int((c0[2] + t * (c1[2] - c0[2])) * brightness),
            )

            size = max(1, int(p.radius * perspective * 1.8))

            pygame.draw.circle(surface, color, (sx, sy), size)

# =============================================================================
# APPLICATION
# =============================================================================

class App:
    def __init__(self):
        log.info("Initialising Pygame...")
        pygame.init()

        info   = pygame.display.Info()
        self.W = info.current_w
        self.H = info.current_h
        log.info("Display: %dx%d", self.W, self.H)

        try:
            self.screen = pygame.display.set_mode((self.W, self.H), pygame.NOFRAME)
        except pygame.error:
            self.W, self.H = 1280, 720
            self.screen = pygame.display.set_mode((self.W, self.H))

        pygame.display.set_caption("Particle Galaxy")

        self.trail   = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        self.trail.fill((0, 0, 0, TRAIL_ALPHA))

        self.clock   = pygame.time.Clock()
        self.font    = pygame.font.SysFont("consolas", 18)
        self.t_start = time.time()

        self.swarm   = ParticleSwarm(NUM_PARTICLES, self.W, self.H)
        self.tracker = HandTracker()
        self.tracker.start()
        log.info("Ready.")

    def run(self):
        running     = True
        palette_idx = 0
        pal_timer   = 0.0
        PAL_INTERVAL = 8.0

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            t_global   = time.time() - self.t_start
            pal_timer += 1.0 / FPS_TARGET
            if pal_timer >= PAL_INTERVAL:
                pal_timer   = 0.0
                palette_idx = (palette_idx + 1) % len(PALETTES)

            state   = self.tracker.get_state()
            palette = PALETTES[palette_idx]

            self.swarm.update(state, t_global)

            self.screen.blit(self.trail, (0, 0))
            self.swarm.draw(self.screen, palette, t_global)

            # Glow ring at palm
            if state is not None:
                px     = int(state.palm_x * self.W)
                py     = int(state.palm_y * self.H)
                ring_r = int(40 + state.spread * 80) if state.is_open else 20
                alpha  = 60 if state.is_open else 120
                glow   = pygame.Surface((ring_r * 2 + 4, ring_r * 2 + 4), pygame.SRCALPHA)
                pygame.draw.circle(glow, (palette[1][0], palette[1][1], palette[1][2], alpha),
                                   (ring_r + 2, ring_r + 2), ring_r, 2)
                self.screen.blit(glow, (px - ring_r - 2, py - ring_r - 2))

            # HUD
            if state is None:
                hand_str = "no hand"
            elif state.is_open:
                hand_str = "open hand"
            else:
                hand_str = "fist closed"
            hud = "FPS: %d   Hand: %s" % (int(self.clock.get_fps()), hand_str)
            self.screen.blit(self.font.render(hud, True, (140, 140, 160)), (14, 14))
            self.screen.blit(self.font.render("ESC to exit", True, (60, 60, 70)), (14, self.H - 28))

            pygame.display.flip()
            self.clock.tick(FPS_TARGET)

        self._quit()

    def _quit(self):
        log.info("Shutting down...")
        self.tracker.stop()
        pygame.quit()
        sys.exit(0)

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    try:
        App().run()
    except Exception:
        log.critical("FATAL ERROR:\n%s", traceback.format_exc())
        input("\nSee swarm_log.txt for details.\nPress Enter to exit...")
        sys.exit(1)