"""
NSFW Content Filter — Video Frame Sampling Strategy

Extracts keyframes from videos for classification:
    1. Scene-change detection via histogram comparison
    2. Uniform temporal sampling as fallback (1 FPS)
    3. Each keyframe is passed through the image model
    4. Video flagged as NSFW if ANY keyframe exceeds threshold
"""

import logging
from pathlib import Path
from typing import Generator, Iterator, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_SCENE_THRESHOLD = 0.4   # Histogram difference threshold for scene change
DEFAULT_UNIFORM_FPS = 1.0       # Frames per second for uniform sampling
MAX_KEYFRAMES = 20              # Cap to avoid processing extremely long videos
MIN_FRAME_INTERVAL = 10         # Minimum frames between keyframes


# ===========================================================================
# Scene-Change Detection
# ===========================================================================

class SceneChangeDetector:
    """
    Detects scene changes by comparing histograms of consecutive frames.

    Uses the correlation method on HSV color histograms.
    When the correlation drops below the threshold, a scene change
    is flagged and the new frame is captured as a keyframe.
    """

    def __init__(self, threshold: float = DEFAULT_SCENE_THRESHOLD):
        self.threshold = threshold

    @staticmethod
    def _compute_histogram(frame: np.ndarray) -> np.ndarray:
        """Compute normalized HSV histogram for a frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1], None, [50, 60], [0, 180, 0, 256]
        )
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def detect(self, video_path: str) -> Generator[np.ndarray, None, None]:
        """
        Yield keyframes at scene-change boundaries one by one.

        Args:
            video_path: Path to the video file.

        Yields:
            Keyframe images (as BGR numpy arrays).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Cannot open video: %s", video_path)
            return

        prev_hist = None
        frame_idx = 0
        last_keyframe_idx = -MIN_FRAME_INTERVAL  # ensure first frame is captured
        yielded_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            curr_hist = self._compute_histogram(frame)

            if prev_hist is None:
                # Always capture the first frame
                yield frame.copy()
                yielded_count += 1
                last_keyframe_idx = frame_idx
            else:
                # Compare with previous histogram
                correlation = cv2.compareHist(
                    prev_hist, curr_hist, cv2.HISTCMP_CORREL
                )
                # Low correlation → scene change
                if (
                    correlation < self.threshold
                    and (frame_idx - last_keyframe_idx) >= MIN_FRAME_INTERVAL
                ):
                    yield frame.copy()
                    yielded_count += 1
                    last_keyframe_idx = frame_idx

            prev_hist = curr_hist
            frame_idx += 1

            if yielded_count >= MAX_KEYFRAMES:
                logger.info("Max keyframes (%d) reached", MAX_KEYFRAMES)
                break

        cap.release()
        logger.info(
            "Scene-change detection: Yielded %d keyframes from %d frames (%s)",
            yielded_count, frame_idx, video_path,
        )


# ===========================================================================
# Uniform Temporal Sampling
# ===========================================================================

class UniformSampler:
    """
    Samples frames at a fixed rate (e.g., 1 FPS).
    Used as a simpler alternative or fallback if scene detection
    produces too few frames.
    """

    def __init__(self, target_fps: float = DEFAULT_UNIFORM_FPS):
        self.target_fps = target_fps

    def sample(self, video_path: str) -> Generator[np.ndarray, None, None]:
        """
        Yield frames at a uniform temporal rate one by one.

        Args:
            video_path: Path to the video file.

        Yields:
            Sampled frame images (BGR numpy arrays).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Cannot open video: %s", video_path)
            return

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30.0  # fallback

        # Sample every N-th frame
        sample_interval = max(1, int(video_fps / self.target_fps))

        frame_idx = 0
        yielded_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval == 0:
                yield frame.copy()
                yielded_count += 1

            frame_idx += 1

            if yielded_count >= MAX_KEYFRAMES:
                break

        cap.release()
        logger.info(
            "Uniform sampling (%.1f FPS): Yielded %d frames from %d total (%s)",
            self.target_fps, yielded_count, frame_idx, video_path,
        )


# ===========================================================================
# Combined Video Sampler
# ===========================================================================

class VideoFrameSampler:
    """
    Combined sampler: tries scene-change detection first,
    falls back to uniform sampling if too few keyframes are found.

    Usage:
        sampler = VideoFrameSampler()
        for f in sampler.extract_keyframes("video.mp4"):
            ...
    """

    def __init__(
        self,
        scene_threshold: float = DEFAULT_SCENE_THRESHOLD,
        uniform_fps: float = DEFAULT_UNIFORM_FPS,
        min_keyframes: int = 3,
    ):
        self.scene_detector = SceneChangeDetector(threshold=scene_threshold)
        self.uniform_sampler = UniformSampler(target_fps=uniform_fps)
        self.min_keyframes = min_keyframes

    def extract_keyframes(self, video_path: str) -> Generator[Image.Image, None, None]:
        """
        Yield keyframes from a video file dynamically to minimize memory.

        Strategy:
            1. Try scene-change detection
            2. If < min_keyframes found, fall back to uniform sampling
            3. Convert all frames to PIL Images (RGB) and resize dynamically

        Args:
            video_path: Path to the video file.

        Yields:
            PIL Image objects (RGB).
        """
        # We need to know if scene detection yielded enough frames.
        # But we don't want to load all frames into memory to find out.
        # So we do a fast pass just to count scene changes, without copying frames.
        scene_count = 0
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            prev_h = None
            f_idx = 0
            last_k = -MIN_FRAME_INTERVAL
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                # Fast histogram
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
                cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                
                if prev_h is None:
                    scene_count += 1
                    last_k = f_idx
                else:
                    corr = cv2.compareHist(prev_h, hist, cv2.HISTCMP_CORREL)
                    if corr < self.scene_detector.threshold and (f_idx - last_k) >= MIN_FRAME_INTERVAL:
                        scene_count += 1
                        last_k = f_idx
                prev_h = hist
                f_idx += 1
                if scene_count >= MAX_KEYFRAMES: break
            cap.release()

        use_scene = scene_count >= self.min_keyframes
        if not use_scene:
            logger.info("Scene detection yielded %d frames (< %d), falling back to uniform sampling", scene_count, self.min_keyframes)

        generator = self.scene_detector.detect(video_path) if use_scene else self.uniform_sampler.sample(video_path)

        # Convert BGR (OpenCV) → RGB (PIL) and resize to save memory
        max_width = 640
        yielded_count = 0
        for frame in generator:
            h, w = frame.shape[:2]
            if w > max_width:
                scale = max_width / w
                new_size = (max_width, int(h * scale))
                frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield Image.fromarray(rgb)
            yielded_count += 1

        logger.info("Total dynamic keyframes yielded: %d (resized to max %dpx) from %s", yielded_count, max_width, video_path)

    @staticmethod
    def get_video_info(video_path: str) -> dict:
        """Get basic video metadata."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": f"Cannot open {video_path}"}

        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration_seconds": (
                cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1)
            ),
        }
        cap.release()
        return info


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python video_sampler.py <video_path>")
        sys.exit(1)

    video_file = sys.argv[1]
    print(f"\n--- Video Info ---")
    info = VideoFrameSampler.get_video_info(video_file)
    for k, v in info.items():
        print(f"  {k}: {v}")

    sampler = VideoFrameSampler()
    keyframes = sampler.extract_keyframes(video_file)
    print(f"\nExtracted {len(keyframes)} keyframes")

    # Save keyframes for inspection
    out_dir = Path("keyframes_debug")
    out_dir.mkdir(exist_ok=True)
    for i, img in enumerate(keyframes):
        img.save(out_dir / f"keyframe_{i:03d}.jpg")
    print(f"Saved to {out_dir}/")
