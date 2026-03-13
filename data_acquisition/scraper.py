"""
NSFW Content Filter — Data Acquisition: Web Scraper

Strategy for scraping a balanced dataset of 'Safe' vs. 'NSFW' content
from the internet. Uses public APIs (Reddit, Flickr) for ethical sourcing.

IMPORTANT: This module provides the STRATEGY and code structure.
    You must configure API keys and comply with Terms of Service
    before running any actual scraping.
"""

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Curated source lists — expand as needed
SAFE_SUBREDDITS = [
    "EarthPorn", "FoodPorn", "CityPorn", "ArchitecturePorn",
    "NaturePics", "aww", "pics", "itookapicture",
]
NSFW_SUBREDDITS = [
    "nsfw", "gonewild", "RealGirls"
]

FLICKR_SAFE_TAGS = ["landscape", "architecture", "food", "nature", "city"]
FLICKR_NSFW_TAGS: List[str] = []  # Flickr SafeSearch off — configure carefully

DEFAULT_HEADERS = {
    "User-Agent": (
        "NSFWFilterResearchBot/1.0 "
        "(Academic research; contact@example.com)"
    ),
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _file_hash(content: bytes) -> str:
    """Return SHA-256 hex digest for duplicate detection."""
    return hashlib.sha256(content).hexdigest()


def _is_valid_image_url(url: str) -> bool:
    """Quick heuristic to check if a URL points to an image."""
    parsed = urlparse(url)
    return any(
        parsed.path.lower().endswith(ext)
        for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    )


# ---------------------------------------------------------------------------
# Reddit Scraper
# ---------------------------------------------------------------------------

class RedditImageScraper:
    """
    Scrapes image URLs from Reddit using the public JSON API.

    Reddit provides a `.json` suffix on any listing URL, which
    returns structured data without needing OAuth for read-only
    public subreddits.
    """

    BASE_URL = "https://www.reddit.com/r/{subreddit}/{sort}.json"

    def __init__(
        self,
        output_dir: str = "data",
        images_per_subreddit: int = 500,
        rate_limit_seconds: float = 2.0,
    ):
        self.output_dir = Path(output_dir)
        self.images_per_subreddit = images_per_subreddit
        self.rate_limit = rate_limit_seconds
        self.seen_hashes: Set[str] = set()
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)

    # ------------------------------------------------------------------

    def scrape_subreddit(
        self,
        subreddit: str,
        label: str,
        sort: str = "top",
        time_filter: str = "all",
    ) -> int:
        """
        Download images from a single subreddit.

        Args:
            subreddit: Subreddit name (without r/).
            label: 'safe' or 'nsfw' — determines output subdirectory.
            sort: Reddit sort mode (top, hot, new).
            time_filter: Time filter for 'top' sort (all, year, month).

        Returns:
            Number of images successfully downloaded.
        """
        save_dir = self.output_dir / label
        save_dir.mkdir(parents=True, exist_ok=True)

        url = self.BASE_URL.format(subreddit=subreddit, sort=sort)
        params = {"limit": 100, "t": time_filter, "raw_json": 1}
        after = None
        downloaded = 0

        logger.info(
            "Scraping r/%s (%s) — target: %d images",
            subreddit, label, self.images_per_subreddit,
        )

        while downloaded < self.images_per_subreddit:
            if after:
                params["after"] = after

            try:
                resp = self.session.get(url, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
            except (requests.RequestException, ValueError) as exc:
                logger.warning("Request failed for r/%s: %s", subreddit, exc)
                break

            posts = data.get("data", {}).get("children", [])
            if not posts:
                logger.info("No more posts in r/%s", subreddit)
                break

            for post in posts:
                post_data = post.get("data", {})
                image_url = post_data.get("url", "")

                if not _is_valid_image_url(image_url):
                    continue

                count = self._download_image(image_url, save_dir)
                downloaded += count

                if downloaded >= self.images_per_subreddit:
                    break

            after = data.get("data", {}).get("after")
            if not after:
                break

            time.sleep(self.rate_limit)

        logger.info(
            "Finished r/%s — downloaded %d images", subreddit, downloaded
        )
        return downloaded

    # ------------------------------------------------------------------

    def _download_image(self, url: str, save_dir: Path) -> int:
        """Download a single image, skipping duplicates. Returns 1 or 0."""
        try:
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
            content = resp.content

            # Duplicate detection
            h = _file_hash(content)
            if h in self.seen_hashes:
                return 0
            self.seen_hashes.add(h)

            # Determine extension
            ext = Path(urlparse(url).path).suffix or ".jpg"
            filename = f"{h[:16]}{ext}"
            filepath = save_dir / filename

            filepath.write_bytes(content)
            return 1

        except requests.RequestException as exc:
            logger.debug("Failed to download %s: %s", url, exc)
            return 0

    # ------------------------------------------------------------------

    def run(self) -> Dict[str, int]:
        """Run the scraper for all configured subreddits."""
        stats: Dict[str, int] = {"safe": 0, "nsfw": 0}

        for sub in SAFE_SUBREDDITS:
            stats["safe"] += self.scrape_subreddit(sub, "safe")

        for sub in NSFW_SUBREDDITS:
            stats["nsfw"] += self.scrape_subreddit(sub, "nsfw")

        logger.info("Scraping complete — %s", stats)
        return stats


# ---------------------------------------------------------------------------
# Flickr Scraper (requires API key)
# ---------------------------------------------------------------------------

class FlickrImageScraper:
    """
    Scrapes images from Flickr using their public API.

    Requires a free Flickr API key:
    https://www.flickr.com/services/api/misc.api_keys.html
    """

    SEARCH_URL = "https://api.flickr.com/services/rest/"

    def __init__(
        self,
        api_key: str,
        output_dir: str = "data",
        images_per_tag: int = 500,
        rate_limit_seconds: float = 1.0,
    ):
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.images_per_tag = images_per_tag
        self.rate_limit = rate_limit_seconds
        self.seen_hashes: Set[str] = set()
        self.session = requests.Session()

    def search_and_download(
        self,
        tag: str,
        label: str,
        safe_search: int = 1,
    ) -> int:
        """
        Search Flickr by tag and download images.

        Args:
            tag: Search tag.
            label: 'safe' or 'nsfw'.
            safe_search: 1 = safe, 2 = moderate, 3 = restricted.

        Returns:
            Number of images downloaded.
        """
        save_dir = self.output_dir / label
        save_dir.mkdir(parents=True, exist_ok=True)

        downloaded = 0
        page = 1

        while downloaded < self.images_per_tag:
            params = {
                "method": "flickr.photos.search",
                "api_key": self.api_key,
                "tags": tag,
                "safe_search": safe_search,
                "content_type": 1,  # photos only
                "media": "photos",
                "per_page": 100,
                "page": page,
                "format": "json",
                "nojsoncallback": 1,
                "extras": "url_m",
            }

            try:
                resp = self.session.get(
                    self.SEARCH_URL, params=params, timeout=15
                )
                resp.raise_for_status()
                data = resp.json()
            except (requests.RequestException, ValueError) as exc:
                logger.warning("Flickr API error for tag '%s': %s", tag, exc)
                break

            photos = data.get("photos", {}).get("photo", [])
            if not photos:
                break

            for photo in photos:
                image_url = photo.get("url_m")
                if not image_url:
                    continue

                try:
                    img_resp = self.session.get(image_url, timeout=10)
                    img_resp.raise_for_status()
                    content = img_resp.content

                    h = _file_hash(content)
                    if h in self.seen_hashes:
                        continue
                    self.seen_hashes.add(h)

                    ext = Path(urlparse(image_url).path).suffix or ".jpg"
                    filepath = save_dir / f"{h[:16]}{ext}"
                    filepath.write_bytes(content)
                    downloaded += 1

                except requests.RequestException:
                    continue

                if downloaded >= self.images_per_tag:
                    break

            page += 1
            time.sleep(self.rate_limit)

        logger.info("Flickr tag '%s' — downloaded %d images", tag, downloaded)
        return downloaded

    def run(self) -> Dict[str, int]:
        """Run Flickr scraper for configured tags."""
        stats: Dict[str, int] = {"safe": 0, "nsfw": 0}

        for tag in FLICKR_SAFE_TAGS:
            stats["safe"] += self.search_and_download(tag, "safe", safe_search=1)

        for tag in FLICKR_NSFW_TAGS:
            stats["nsfw"] += self.search_and_download(
                tag, "nsfw", safe_search=3
            )

        logger.info("Flickr scraping complete — %s", stats)
        return stats


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    """Run all scrapers sequentially."""
    print("=" * 60)
    print("NSFW Content Filter — Data Acquisition")
    print("=" * 60)

    # --- Reddit ---
    reddit_scraper = RedditImageScraper(
        output_dir="data",
        images_per_subreddit=10,
        rate_limit_seconds=2.0,
    )
    reddit_stats = reddit_scraper.run()
    print(f"\nReddit stats: {reddit_stats}")

    # --- Flickr (uncomment and set API key) ---
    # flickr_scraper = FlickrImageScraper(
    #     api_key="YOUR_FLICKR_API_KEY",
    #     output_dir="data",
    #     images_per_tag=500,
    # )
    # flickr_stats = flickr_scraper.run()
    # print(f"\nFlickr stats: {flickr_stats}")


if __name__ == "__main__":
    main()
