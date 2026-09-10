"""
Pure Python WebM-to-Ogg Opus Remuxer.

Enables native decoding of YouTube/WebM audio streams via soundfile/librosa
without requiring external ffmpeg installation.
"""

import io
import struct
from typing import Optional, List, Tuple


def ogg_crc(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= (byte << 24)
        for _ in range(8):
            if crc & 0x80000000:
                crc = ((crc << 1) ^ 0x04C11DB7) & 0xFFFFFFFF
            else:
                crc = (crc << 1) & 0xFFFFFFFF
    return crc


def make_ogg_page(header_type: int, granule_pos: int, serial_no: int, page_seq: int, packets: List[bytes]) -> bytes:
    lacing_vals = []
    payload = bytearray()
    for pkt in packets:
        plen = len(pkt)
        while plen >= 255:
            lacing_vals.append(255)
            plen -= 255
        lacing_vals.append(plen)
        payload.extend(pkt)

    header = bytearray(27 + len(lacing_vals))
    header[0:4] = b"OggS"
    header[4] = 0
    header[5] = header_type
    struct.pack_into("<Q", header, 6, granule_pos)
    struct.pack_into("<I", header, 14, serial_no)
    struct.pack_into("<I", header, 18, page_seq)
    struct.pack_into("<I", header, 22, 0)
    header[26] = len(lacing_vals)
    header[27:27 + len(lacing_vals)] = bytes(lacing_vals)

    full_page = header + payload
    crc = ogg_crc(full_page)
    struct.pack_into("<I", full_page, 22, crc)
    return bytes(full_page)


def get_opus_samples(packet: bytes) -> int:
    """Calculate the number of 48kHz audio samples in an Opus packet from its TOC byte."""
    if not packet:
        return 960  # Default 20ms at 48kHz
    toc = packet[0]
    config = toc >> 3
    c = toc & 0x03

    # Frame duration based on RFC 6716 Table 2
    if config < 12:
        dur_ms = [10, 20, 40, 60][config % 4]
    elif config < 16:
        dur_ms = [10, 20][config % 2]
    else:
        dur_ms = [2.5, 5, 10, 20][config % 4]

    samples_per_frame = int(dur_ms * 48)

    if c == 0:
        return samples_per_frame
    elif c in (1, 2):
        return samples_per_frame * 2
    elif c == 3:
        if len(packet) > 1:
            frame_count = packet[1] & 0x3F
            return samples_per_frame * max(1, frame_count)
        return samples_per_frame
    return 960


def read_vint(stream, is_id=False) -> Tuple[Optional[int], int]:
    first = stream.read(1)
    if not first:
        return None, 0
    b = first[0]
    mask = 0x80
    length = 1
    while not (b & mask):
        mask >>= 1
        length += 1
        if length > 8:
            return None, 0

    rem = stream.read(length - 1)
    if len(rem) < length - 1:
        return None, 0

    val_bytes = first + rem
    if is_id:
        val = int.from_bytes(val_bytes, "big")
    else:
        val = b & (~mask)
        for nb in rem:
            val = (val << 8) | nb
    return val, length


def remux_webm_to_ogg(webm_bytes: bytes) -> bytes:
    """
    Extracts Opus audio packets from WebM and repacks them into standard Ogg Opus bytes.
    """
    stream = io.BytesIO(webm_bytes)
    opus_head: Optional[bytes] = None
    opus_packets: List[bytes] = []
    audio_track_num = 1

    file_size = len(webm_bytes)

    while stream.tell() < file_size:
        elem_id, _ = read_vint(stream, is_id=True)
        if elem_id is None:
            break
        elem_size, _ = read_vint(stream, is_id=False)
        if elem_size is None:
            break

        start_pos = stream.tell()

        # Handle container elements (recurse into them)
        # EBML (0x1A45DFA3), Segment (0x18538067), Tracks (0x1654AE6B),
        # TrackEntry (0xAE), Cluster (0x1F43B675)
        if elem_id in (0x1A45DFA3, 0x18538067, 0x1654AE6B, 0xAE, 0x1F43B675):
            continue

        # CodecPrivate in TrackEntry contains OpusHead
        if elem_id == 0x63A2:
            data = stream.read(elem_size)
            if data.startswith(b"OpusHead"):
                opus_head = data
            elif len(data) >= 19:
                # Some encoders omit literal "OpusHead" prefix
                opus_head = b"OpusHead" + data[8:] if not data.startswith(b"OpusHead") else data

        # SimpleBlock in Cluster
        elif elem_id == 0xA3:
            block_data = stream.read(elem_size)
            if len(block_data) > 4:
                b_stream = io.BytesIO(block_data)
                track_num, _ = read_vint(b_stream, is_id=False)
                # Skip 2 bytes timecode + 1 byte flags
                b_stream.seek(3, io.SEEK_CUR)
                packet = b_stream.read()
                if packet:
                    opus_packets.append(packet)

        else:
            # Skip element payload
            stream.seek(start_pos + elem_size)

    if not opus_packets:
        raise ValueError("Could not find any Opus audio packets in WebM stream.")

    if not opus_head:
        # Standard fallback OpusHead: 48000Hz, 2 channels, 0 preskip
        # Magic (8b) + Version (1b) + Channels (1b) + Preskip (2b) + SampleRate (4b) + Gain (2b) + Mapping (1b)
        opus_head = struct.pack("<8sBBHIhB", b"OpusHead", 1, 2, 0, 48000, 0, 0)

    # Standard OpusTags comment
    vendor = b"deepfake-detector-2.0"
    opus_tags = struct.pack("<8sI", b"OpusTags", len(vendor)) + vendor + struct.pack("<I", 0)

    ogg_output = bytearray()
    serial_no = 0x4446414B  # "DFAK"

    # Page 0: OpusHead (BOS: type=2, granule=0, seq=0)
    ogg_output.extend(make_ogg_page(2, 0, serial_no, 0, [opus_head]))

    # Page 1: OpusTags (type=0, granule=0, seq=1)
    ogg_output.extend(make_ogg_page(0, 0, serial_no, 1, [opus_tags]))

    # Subsequent pages: Audio packets (accumulate ~1-2 packets per page for low latency)
    seq = 2
    granule = 0
    page_pkts = []
    page_samples = 0

    for i, pkt in enumerate(opus_packets):
        is_last = (i == len(opus_packets) - 1)
        pkt_samples = get_opus_samples(pkt)
        page_pkts.append(pkt)
        page_samples += pkt_samples

        # Emit page if accumulated > 50 packets or if last packet
        if len(page_pkts) >= 20 or is_last:
            granule += page_samples
            page_type = 4 if is_last else 0  # 4 = EOS
            ogg_output.extend(make_ogg_page(page_type, granule, serial_no, seq, page_pkts))
            seq += 1
            page_pkts = []
            page_samples = 0

    return bytes(ogg_output)
