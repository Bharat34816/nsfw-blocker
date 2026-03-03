"""
NSFW Content Filter — Data Cleaning & Balancing

Validates downloaded images, removes corrupt/tiny files,
enforces class balance, and creates train/val/test splits.
"""

import hashlib
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

MIN_IMAGE_SIZE = (64, 64)       # Minimum width × height
MAX_IMAGE_SIZE = (4096, 4096)   # Maximum width × height
MIN_FILE_SIZE_KB = 5            # Minimum file size in KB
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train / val / test


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class ImageValidator:
    """Validates individual images for quality and integrity."""

    def __init__(
        self,
        min_size: Tuple[int, int] = MIN_IMAGE_SIZE,
        max_size: Tuple[int, int] = MAX_IMAGE_SIZE,
        min_file_kb: int = MIN_FILE_SIZE_KB,
    ):
        self.min_size = min_size
        self.max_size = max_size
        self.min_file_kb = min_file_kb

    def is_valid(self, filepath: Path) -> bool:
        """
        Check if an image file is valid.

        Criteria:
            1. File extension is in VALID_EXTENSIONS
            2. File size > min_file_kb
            3. PIL can open and verify the image
            4. Dimensions are within min/max bounds
        """
        # Extension check
        if filepath.suffix.lower() not in VALID_EXTENSIONS:
            return False

        # File size check
        file_size_kb = filepath.stat().st_size / 1024
        if file_size_kb < self.min_file_kb:
            return False

        # PIL verification
        try:
            with Image.open(filepath) as img:
                img.verify()

            # Re-open after verify (verify closes internal state)
            with Image.open(filepath) as img:
                w, h = img.size
                if w < self.min_size[0] or h < self.min_size[1]:
                    return False
                if w > self.max_size[0] or h > self.max_size[1]:
                    return False

        except Exception:
            return False

        return True


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate_directory(directory: Path) -> int:
    """
    Remove duplicate images from a directory based on file hash.

    Returns:
        Number of duplicates removed.
    """
    seen_hashes = set()
    removed = 0

    for filepath in sorted(directory.iterdir()):
        if not filepath.is_file():
            continue

        file_hash = hashlib.md5(filepath.read_bytes()).hexdigest()

        if file_hash in seen_hashes:
            filepath.unlink()
            removed += 1
        else:
            seen_hashes.add(file_hash)

    logger.info("Deduplication: removed %d duplicates from %s", removed, directory)
    return removed


# ---------------------------------------------------------------------------
# Class Balancing
# ---------------------------------------------------------------------------

def balance_classes(
    safe_dir: Path,
    nsfw_dir: Path,
    strategy: str = "undersample",
) -> Dict[str, int]:
    """
    Balance the Safe and NSFW classes.

    Args:
        safe_dir: Directory containing safe images.
        nsfw_dir: Directory containing NSFW images.
        strategy: 'undersample' (reduce majority) or 'oversample' (duplicate minority).

    Returns:
        Dict with final counts per class.
    """
    safe_files = sorted([f for f in safe_dir.iterdir() if f.is_file()])
    nsfw_files = sorted([f for f in nsfw_dir.iterdir() if f.is_file()])

    safe_count = len(safe_files)
    nsfw_count = len(nsfw_files)

    logger.info(
        "Class counts before balancing — Safe: %d, NSFW: %d",
        safe_count, nsfw_count,
    )

    if strategy == "undersample":
        target = min(safe_count, nsfw_count)

        if safe_count > target:
            to_remove = random.sample(safe_files, safe_count - target)
            for f in to_remove:
                f.unlink()

        if nsfw_count > target:
            to_remove = random.sample(nsfw_files, nsfw_count - target)
            for f in to_remove:
                f.unlink()

    elif strategy == "oversample":
        target = max(safe_count, nsfw_count)

        minority_dir = safe_dir if safe_count < nsfw_count else nsfw_dir
        minority_files = safe_files if safe_count < nsfw_count else nsfw_files
        deficit = target - len(minority_files)

        for i in range(deficit):
            src = random.choice(minority_files)
            ext = src.suffix
            dst = minority_dir / f"oversample_{i:05d}{ext}"
            shutil.copy2(src, dst)

    # Recount
    final_safe = len([f for f in safe_dir.iterdir() if f.is_file()])
    final_nsfw = len([f for f in nsfw_dir.iterdir() if f.is_file()])

    logger.info(
        "Class counts after balancing — Safe: %d, NSFW: %d",
        final_safe, final_nsfw,
    )
    return {"safe": final_safe, "nsfw": final_nsfw}


# ---------------------------------------------------------------------------
# Train / Val / Test Splitting
# ---------------------------------------------------------------------------

def create_splits(
    source_dir: Path,
    output_dir: Path,
    ratios: Tuple[float, float, float] = SPLIT_RATIOS,
    seed: int = 42,
) -> Dict[str, Dict[str, int]]:
    """
    Split data into train/val/test sets, preserving class directories.

    Expects:
        source_dir/
            safe/   ← image files
            nsfw/   ← image files

    Creates:
        output_dir/
            train/safe/  train/nsfw/
            val/safe/    val/nsfw/
            test/safe/   test/nsfw/

    Returns:
        Nested dict with counts per split per class.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1.0"

    random.seed(seed)
    splits = ("train", "val", "test")
    stats: Dict[str, Dict[str, int]] = {s: {} for s in splits}

    for label in ("safe", "nsfw"):
        class_dir = source_dir / label
        if not class_dir.exists():
            logger.warning("Missing class directory: %s", class_dir)
            continue

        files = sorted([f for f in class_dir.iterdir() if f.is_file()])
        random.shuffle(files)

        n = len(files)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        # Rest goes to test
        partitions = {
            "train": files[:n_train],
            "val": files[n_train : n_train + n_val],
            "test": files[n_train + n_val :],
        }

        for split_name, split_files in partitions.items():
            dest_dir = output_dir / split_name / label
            dest_dir.mkdir(parents=True, exist_ok=True)

            for f in split_files:
                shutil.copy2(f, dest_dir / f.name)

            stats[split_name][label] = len(split_files)
            logger.info(
                "Split %s/%s — %d files", split_name, label, len(split_files)
            )

    return stats


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def clean_and_split(
    raw_dir: str = "data",
    processed_dir: str = "data_processed",
    balance_strategy: str = "undersample",
) -> Dict:
    """
    Run the complete cleaning pipeline:
        1. Validate images (remove corrupt/tiny)
        2. Deduplicate
        3. Balance classes
        4. Create train/val/test splits
    """
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)

    validator = ImageValidator()
    report: Dict = {"removed_invalid": 0, "removed_duplicates": 0}

    # --- Step 1: Validate ---
    for label in ("safe", "nsfw"):
        class_dir = raw_path / label
        if not class_dir.exists():
            continue

        for filepath in list(class_dir.iterdir()):
            if filepath.is_file() and not validator.is_valid(filepath):
                filepath.unlink()
                report["removed_invalid"] += 1

    logger.info("Removed %d invalid images", report["removed_invalid"])

    # --- Step 2: Deduplicate ---
    for label in ("safe", "nsfw"):
        class_dir = raw_path / label
        if class_dir.exists():
            report["removed_duplicates"] += deduplicate_directory(class_dir)

    # --- Step 3: Balance ---
    safe_dir = raw_path / "safe"
    nsfw_dir = raw_path / "nsfw"
    if safe_dir.exists() and nsfw_dir.exists():
        report["balance"] = balance_classes(
            safe_dir, nsfw_dir, strategy=balance_strategy
        )

    # --- Step 4: Split ---
    report["splits"] = create_splits(raw_path, processed_path)

    logger.info("Pipeline complete — Report: %s", report)
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("NSFW Content Filter — Data Cleaning & Splitting")
    print("=" * 60)
    result = clean_and_split()
    print(f"\nFinal report: {result}")
