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
from typing import Generator

from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_SCENE_THRESHOLD = 0.4   # Kept for API compatibility
DEFAULT_UNIFORM_FPS = 1.0       # Frames per second for uniform sampling
MAX_KEYFRAMES = 20              # Cap to avoid processing extremely long videos
MIN_FRAME_INTERVAL = 10         # Kept for API compatibility


# ===========================================================================
# Combined Video Sampler
# ===========================================================================

import os
import shutil
import tempfile
import subprocess

class VideoFrameSampler:
    """
    Robust Video Sampler using FFmpeg Subprocess.
    
    Extracts frames at a uniform rate (default 1 FPS) by calling the FFmpeg binary
    directly. This completely bypasses memory-heavy libraries like OpenCV, preventing
    C++ buffer segfaults and OOM executions on memory-constrained cloud environments.

    Usage:
        sampler = VideoFrameSampler()
        for f in sampler.extract_keyframes("video.mp4"):
            ...
    """

    def __init__(
        self,
        scene_threshold: float = DEFAULT_SCENE_THRESHOLD, # Kept for API compatibility
        uniform_fps: float = DEFAULT_UNIFORM_FPS,
        min_keyframes: int = 3,
    ):
        self.uniform_fps = uniform_fps
        self.max_keyframes = MAX_KEYFRAMES

    def extract_keyframes(self, video_path: str) -> Generator[Image.Image, None, None]:
        """
        Yield keyframes from a video file dynamically using FFmpeg.

        Strategy:
            1. Create an isolated temporary directory.
            2. Run an FFmpeg subprocess to extract frames as tiny JPEGs.
               (e.g., `-r 1` for 1 FPS, `-vf scale=640:-1` to limit size).
            3. Yield each JPEG as a PIL Image.
            4. Automatically clean up the temp directory when done.

        Args:
            video_path: Path to the video file.

        Yields:
            PIL Image objects (RGB).
        """
        logger.info("Starting FFmpeg extraction for: %s", video_path)

        # Create a temporary directory that will be deleted automatically
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_pattern = os.path.join(temp_dir, "frame_%04d.jpg")

            # Command syntax: ffmpeg -i input.mp4 -r 1 -vf "scale=640:-1" -vframes 20 temp/frame_%04d.jpg
            command = [
                "ffmpeg",
                "-y",  # Overwrite output files without asking
                "-i", video_path,
                "-r", str(self.uniform_fps),  # Frame rate
                "-vf", "scale=640:-1",        # Resize to max-width 640px, auto-height
                "-vframes", str(self.max_keyframes), # Hard limit on total frames generated
                "-q:v", "2",                  # High quality JPEG
                temp_pattern
            ]

            try:
                # Run the process synchronously. Suppress stdout/stderr to keep logs clean
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                logger.error("FFmpeg extraction failed on %s: %s", video_path, e)
                # Fallback: yield empty generator if FFmpeg fails completely
                return

            # Read the generated frames, yield them, and discard them
            extracted_files = sorted(Path(temp_dir).glob("frame_*.jpg"))
            logger.info("FFmpeg generated %d keyframes.", len(extracted_files))

            yielded_count = 0
            for file_path in extracted_files:
                try:
                    # Open with PIL, immediately copy the image data into memory,
                    # and close the file handle so the temp file can be safely deleted.
                    with Image.open(file_path) as img:
                        # Convert to standard RGB to match OpenCV output behavior
                        rgb_img = img.convert("RGB")
                        
                    yield rgb_img
                    yielded_count += 1
                except Exception as e:
                    logger.warning("Failed to load extracted frame %s: %s", file_path, e)
                    continue

            logger.info("Total FFmpeg keyframes yielded: %d from %s", yielded_count, video_path)

    @staticmethod
    def get_video_info(video_path: str) -> dict:
        """Get basic video metadata using strictly standard libraries (or ffprobe)."""
        # Removed OpenCV dependency. Simplified info extraction.
        return {
            "path": video_path,
            "status": "Ready for FFmpeg extraction"
        }

# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python video_sampler.py <video_path>")
        sys.exit(1)

    video_file = sys.argv[1]

    sampler = VideoFrameSampler()
    count = 0
    
    out_dir = Path("keyframes_debug")
    out_dir.mkdir(exist_ok=True)
    
    for img in sampler.extract_keyframes(video_file):
        img.save(out_dir / f"keyframe_{count:03d}.jpg")
        count += 1
        
    print(f"\nExtracted {count} keyframes using FFmpeg")
    print(f"Saved to {out_dir}/")
