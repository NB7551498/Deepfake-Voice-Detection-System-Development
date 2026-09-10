"""
Unit tests for Pure Python WebM-to-Ogg Opus remuxer and URL safety verification.
"""

import pytest
from src.api.webm_to_ogg import ogg_crc, make_ogg_page, get_opus_samples, remux_webm_to_ogg
from src.api.url_downloader import is_safe_url


def test_ogg_crc():
    """Verify Ogg CRC-32 generator creates non-zero CRC for test string."""
    data = b"OggS\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x04test"
    crc = ogg_crc(data)
    assert isinstance(crc, int)
    assert crc != 0


def test_make_ogg_page():
    """Verify Ogg page structure conforms to RFC 3533 header format."""
    page = make_ogg_page(header_type=2, granule_pos=0, serial_no=12345, page_seq=0, packets=[b"OpusHead1234"])
    assert page.startswith(b"OggS")
    assert len(page) > 27  # Header is 27 bytes + lacing table + payload


def test_get_opus_samples():
    """Verify calculation of 48kHz audio sample counts from Opus TOC byte."""
    # TOC with config 0 (Silk NB 10ms -> 480 samples at 48kHz)
    samples = get_opus_samples(bytes([0b00000000]))
    assert samples == 480

    # TOC with config 12 (CELT 10ms -> 480 samples)
    samples = get_opus_samples(bytes([0b01100000]))
    assert samples == 480


def test_remux_invalid_webm():
    """Verify remuxer raises ValueError gracefully on corrupt or non-Opus stream."""
    with pytest.raises(ValueError):
        remux_webm_to_ogg(b"\x1a\x45\xdf\xa3\x93invalid_webm_data_stream")


def test_is_safe_url_blocks_ssrf():
    """Verify SSRF protection rejects private, loopback, and cloud metadata addresses."""
    unsafe_urls = [
        "http://127.0.0.1/audio.wav",
        "http://localhost:8080/test.mp3",
        "http://169.254.169.254/latest/meta-data/",
        "http://192.168.1.1/secret.wav",
        "ftp://example.com/audio.wav",
        "javascript:alert(1)",
        "",
    ]
    for url in unsafe_urls:
        safe, err = is_safe_url(url)
        assert safe is False, f"Expected {url} to be blocked by SSRF check"
        assert len(err) > 0


def test_is_safe_url_allows_public_web():
    """Verify safe public domains are accepted."""
    safe_urls = [
        "https://www.youtube.com/watch?v=jNQXAC9IVRw",
        "https://raw.githubusercontent.com/Jakobovski/free-spoken-digit-dataset/master/recordings/0_jackson_0.wav",
    ]
    for url in safe_urls:
        safe, err = is_safe_url(url)
        assert safe is True, f"Expected {url} to be allowed, got error: {err}"
