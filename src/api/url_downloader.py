"""
Secure Audio URL Fetcher & Downloader.

Features:
- SSRF Protection: Blocks private IPs, loopback, link-local, AWS/cloud metadata (169.254.169.254)
- Size Limiting: Max 25 MB stream limit
- Format Support: Direct audio URLs (MP3, WAV, M4A, OGG, FLAC) & Platform URLs (YouTube, SoundCloud via yt-dlp)
"""

import io
import ipaddress
import os
import socket
import tempfile
import urllib.parse
from pathlib import Path
import requests

MAX_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB
DOWNLOAD_TIMEOUT = 25  # seconds


def is_safe_url(url: str) -> tuple[bool, str]:
    """Validate that the URL is not pointing to private/internal infrastructure."""
    if not url or not isinstance(url, str):
        return False, "URL string is empty or invalid."

    url = url.strip()
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception as e:
        return False, f"Malformed URL: {str(e)}"

    if parsed.scheme not in ("http", "https"):
        return False, f"Unsupported URL protocol: '{parsed.scheme}'. Only HTTP and HTTPS are permitted."

    hostname = parsed.hostname
    if not hostname:
        return False, "Invalid URL: missing hostname."

    # Prevent localhost explicitly
    if hostname.lower() in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
        return False, "Access to localhost or loopback addresses is forbidden."

    try:
        addr_info = socket.getaddrinfo(hostname, None)
        for entry in addr_info:
            ip_str = entry[4][0]
            ip_obj = ipaddress.ip_address(ip_str)
            if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_multicast or ip_obj.is_reserved:
                return False, f"Access to private or restricted IP address ({ip_str}) is prohibited."
    except Exception as e:
        return False, f"Could not resolve host: {str(e)}"

    return True, ""


def clean_media_url(url: str) -> str:
    """Clean video URLs, stripping playlist, mix, radio, and tracking parameters."""
    url = url.strip()
    try:
        parsed = urllib.parse.urlparse(url)
        netloc = parsed.netloc.lower()
        if "youtu.be" in netloc:
            vid_id = parsed.path.strip("/")
            if vid_id:
                return f"https://www.youtube.com/watch?v={vid_id}"
        elif "youtube.com" in netloc or "music.youtube.com" in netloc:
            if "/shorts/" in parsed.path:
                parts = parsed.path.split("/shorts/")
                if len(parts) > 1:
                    vid_id = parts[1].split("/")[0].split("?")[0]
                    return f"https://www.youtube.com/watch?v={vid_id}"
            qs = urllib.parse.parse_qs(parsed.query)
            v_list = qs.get("v")
            if v_list and v_list[0]:
                return f"https://www.youtube.com/watch?v={v_list[0]}"
    except Exception:
        pass
    return url


def download_audio_from_url(url: str) -> tuple[bytes, str]:
    """
    Download audio from a given URL safely.
    Returns:
        (audio_bytes, filename)
    """
    url = clean_media_url(url.strip())
    is_safe, error_msg = is_safe_url(url)
    if not is_safe:
        raise ValueError(error_msg)

    # Check if URL might be a YouTube or media platform link
    is_media_platform = any(domain in url.lower() for domain in [
        "youtube.com", "youtu.be", "soundcloud.com", "vimeo.com", "tiktok.com", "bilibili.com"
    ])

    if is_media_platform:
        return _download_with_ytdlp(url)

    # Standard Direct HTTP(S) Stream Download
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 DeepfakeDetector/2.0"
    }

    try:
        resp = requests.get(url, headers=headers, stream=True, timeout=DOWNLOAD_TIMEOUT)
        resp.raise_for_status()

        cl = resp.headers.get("content-length")
        if cl and int(cl) > MAX_SIZE_BYTES:
            raise ValueError(f"Audio file size exceeds maximum limit of 25MB (got {int(cl) / (1024*1024):.1f}MB).")

        content = bytearray()
        for chunk in resp.iter_content(chunk_size=65536):
            if chunk:
                content.extend(chunk)
                if len(content) > MAX_SIZE_BYTES:
                    raise ValueError("Audio file size exceeded maximum limit of 25MB while streaming.")

        parsed_path = urllib.parse.urlparse(url).path
        filename = Path(parsed_path).name or "downloaded_audio.wav"
        if not any(filename.lower().endswith(ext) for ext in [".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".webm"]):
            filename += ".wav"

        raw_bytes = bytes(content)
        if filename.lower().endswith(".webm") or raw_bytes.startswith(b"\x1a\x45\xdf\xa3"):
            try:
                from src.api.webm_to_ogg import remux_webm_to_ogg
                return remux_webm_to_ogg(raw_bytes), "downloaded_audio.ogg"
            except Exception:
                pass

        return raw_bytes, filename

    except requests.exceptions.RequestException as e:
        if is_media_platform:
            return _download_with_ytdlp(url)
        raise ValueError(f"Failed to fetch audio from URL: {str(e)}")


def _download_with_ytdlp(url: str) -> tuple[bytes, str]:
    """Extract audio from streaming media URLs via yt-dlp if installed."""
    try:
        import yt_dlp
    except ImportError:
        raise ValueError("Media link detected, but yt-dlp is not available on this server.")

    clean_url = clean_media_url(url)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_template = os.path.join(tmpdir, "audio.%(ext)s")
        ydl_opts = {
            # Prioritize lightweight WebM Opus (249/250/251) which download in seconds and remux natively
            "format": "249/250/251/bestaudio[ext=webm]/bestaudio[ext=ogg]/bestaudio",
            "outtmpl": out_template,
            "max_filesize": MAX_SIZE_BYTES,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "extract_flat": False,
            "nocheckcertificate": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([clean_url])
        except Exception as dl_err:
            raise ValueError(f"Could not download audio from YouTube: {str(dl_err)}")

        files = list(Path(tmpdir).glob("*"))
        if not files:
            raise ValueError("No audio stream could be extracted from this media link.")

        target_file = files[0]
        with open(target_file, "rb") as f:
            data = f.read()

        # If downloaded stream is WebM (Opus container), remux into standard Ogg Opus
        if target_file.suffix.lower() == ".webm" or data.startswith(b"\x1a\x45\xdf\xa3"):
            try:
                from src.api.webm_to_ogg import remux_webm_to_ogg
                ogg_bytes = remux_webm_to_ogg(data)
                return ogg_bytes, "youtube_audio.ogg"
            except Exception as remux_err:
                print(f"[url_downloader] WebM remux warning: {remux_err}")

        return data, target_file.name
