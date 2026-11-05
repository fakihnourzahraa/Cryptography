"""Entropy sources: the real ESP32-CAM over USB serial, and a mock for testing.

Both expose the same methods:
    block(mode)   -> 32 bytes   mode: "E" mixed, "C" camera-only, "H" hardware RNG only
    raw_frame()   -> numpy uint8 array (height, width)
    info()        -> str
"""
from __future__ import annotations

import os
import struct
import time
import zlib

import numpy as np

MAGIC = b"\xA5\x5A"
BLOCK_MODES = ("E", "C", "H")


class DeviceError(RuntimeError):
    pass


def read_packet(stream, timeout: float = 10.0):
    """Read one framed packet from a byte stream with .read(n).

    Skips any junk (boot messages) until the 0xA5 0x5A magic, then reads
    type, length, payload and CRC32. Returns (type_char, payload).
    """
    deadline = time.monotonic() + timeout
    prev = b""
    while True:
        if time.monotonic() > deadline:
            raise DeviceError("timed out waiting for a packet from the ESP32-CAM")
        b = stream.read(1)
        if not b:
            continue
        if prev + b == MAGIC:
            break
        prev = b
    hdr = _read_exact(stream, 5, deadline)
    ptype = chr(hdr[0])
    (length,) = struct.unpack("<I", hdr[1:])
    if length > 4 * 1024 * 1024:
        raise DeviceError(f"implausible packet length {length}")
    payload = _read_exact(stream, length, deadline)
    (crc,) = struct.unpack("<I", _read_exact(stream, 4, deadline))
    if zlib.crc32(payload) != crc:
        raise DeviceError("CRC mismatch: serial data was corrupted")
    if ptype == "!":
        raise DeviceError("device error: " + payload.decode(errors="replace"))
    return ptype, payload


def _read_exact(stream, n: int, deadline: float) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        if time.monotonic() > deadline:
            raise DeviceError(f"timed out: got {len(buf)} of {n} bytes")
        chunk = stream.read(n - len(buf))
        if chunk:
            buf += chunk
    return bytes(buf)


def pick_luma_offset(yuv: np.ndarray) -> int:
    """In YUV422 every other byte is brightness (Y); the rest are colour (U/V).

    Which half is Y depends on the sensor's byte order, so detect it: the Y
    plane has fine detail (big differences between neighbours) and drifts
    away from 128 with the light level, while U/V are smooth and sit near 128.
    The choice is made once per session so all frames use the same plane.
    """
    scores = []
    for off in (0, 1):
        p = yuv[:, off::2].astype(np.int16)
        texture = np.abs(np.diff(p, axis=1)).mean()
        scores.append(texture + 0.25 * abs(p.mean() - 128))
    return int(np.argmax(scores))


def rgb565_green(buf: np.ndarray, h: int, w: int, big_endian: bool) -> np.ndarray:
    """Green channel of an RGB565 frame: the sensor's own 6-bit values (0-63).

    We analyse green rather than a computed grey level because it is the
    sensor's exact output (no rounding from mixing channels), and green has
    the most bits and the most photosites on a Bayer sensor.
    """
    b = buf.reshape(h, w, 2).astype(np.uint16)
    px = (b[..., 0] << 8 | b[..., 1]) if big_endian else (b[..., 1] << 8 | b[..., 0])
    return ((px >> 5) & 0x3F).astype(np.uint8)


def pick_rgb565_big_endian(buf: np.ndarray, h: int, w: int) -> bool:
    """Detect the byte order: decoded the right way, neighbouring pixels are
    similar; decoded the wrong way, the green bits get scrambled. Falls back to
    big-endian (the esp32-camera convention) when the frame is too uniform to
    tell, e.g. with the lens covered."""
    rough = []
    for be in (True, False):
        g = rgb565_green(buf, h, w, be).astype(np.int16)
        rough.append(np.abs(np.diff(g, axis=1)).mean() + 1e-9)
    if rough[1] < rough[0] / 1.5:
        return False
    return True


class CamSource:
    """The real ESP32-CAM, talking the protocol in firmware/entropy_cam."""

    name = "esp32-cam"

    def __init__(self, port: str = "/dev/ttyUSB0", baud: int = 921600, boot_wait: float = 2.5):
        import serial  # pyserial

        env = os.environ.get("PQCAM_LUMA_OFFSET")       # force 0 or 1 if auto-detect guesses wrong
        self.luma_offset = int(env) if env in ("0", "1") else None
        order = os.environ.get("PQCAM_RGB565_ORDER", "").lower()   # "be" or "le" to override
        self.rgb565_be = {"be": True, "le": False}.get(order)

        ser = serial.Serial()
        ser.port = port
        ser.baudrate = baud
        ser.timeout = 0.5
        # The ESP32-CAM-MB wires DTR/RTS to the reset and boot pins. Leaving them
        # asserted when the port opens holds the board in reset or boot mode.
        ser.dtr = False
        ser.rts = False
        ser.open()
        self.ser = ser
        time.sleep(boot_wait)          # let it boot if opening the port reset it
        ser.reset_input_buffer()

    def _cmd(self, c: str, expect: str, timeout: float = 10.0) -> bytes:
        self.ser.write(c.encode())
        ptype, payload = read_packet(self.ser, timeout)
        if ptype != expect:
            raise DeviceError(f"expected packet {expect!r}, got {ptype!r}")
        return payload

    def block(self, mode: str = "E") -> bytes:
        if mode not in BLOCK_MODES:
            raise ValueError(f"mode must be one of {BLOCK_MODES}")
        out = self._cmd(mode, mode, timeout=30.0)    # a stalled capture makes the board restart its camera
        if len(out) != 32:
            raise DeviceError("entropy block has wrong length")
        return out

    def raw_frame(self) -> np.ndarray:
        """One frame as a uint8 (height, width) array of single-channel sensor values.

        Firmware 1.3+ (RGB565): the green channel, values 0-63.
        Older firmware: grayscale or the Y plane of YUV422, values 0-255.
        """
        payload = self._cmd("R", "R", timeout=30.0)
        w, h = struct.unpack("<HH", payload[:4])
        if len(payload) == 6 + 2 * w * h and payload[4] == 3:      # RGB565 (firmware 1.3+)
            data = np.frombuffer(payload[6:], dtype=np.uint8)
            if self.rgb565_be is None:
                self.rgb565_be = pick_rgb565_big_endian(data, h, w)
            return rgb565_green(data, h, w, self.rgb565_be)
        data = np.frombuffer(payload[4:], dtype=np.uint8)
        if data.size == w * h:                       # grayscale (firmware 1.0/1.1)
            return data.reshape(h, w)
        if data.size != 2 * w * h:
            raise DeviceError(f"frame size mismatch: {data.size} bytes for {w}x{h}")
        yuv = data.reshape(h, w * 2)                 # YUV422 (firmware 1.2)
        if self.luma_offset is None:
            self.luma_offset = pick_luma_offset(yuv)
        return np.ascontiguousarray(yuv[:, self.luma_offset::2])

    def info(self) -> str:
        return self._cmd("I", "I").decode(errors="replace")

    def close(self):
        self.ser.close()


class MockCamSource:
    """Stand-in for the ESP32-CAM so the whole pipeline runs without hardware.

    Frames are a fixed synthetic scene plus Gaussian sensor noise; the noise
    comes from os.urandom, so this is NOT a camera entropy source. It exists
    only to test the code. "covered=True" simulates a covered lens (dark frame,
    high gain, noise dominated).
    """

    name = "MOCK (not real camera entropy)"

    def __init__(self, width: int = 320, height: int = 240, covered: bool = False):
        self.w, self.h, self.covered = width, height, covered
        self.rng = np.random.default_rng(int.from_bytes(os.urandom(16), "little"))
        y, x = np.mgrid[0:height, 0:width]
        self.scene = (60 + 120 * (x / width) + 40 * np.sin(y / 15.0)).astype(np.float64)
        self.counter = 0

    def raw_frame(self) -> np.ndarray:
        if self.covered:
            frame = 8 + self.rng.normal(0, 3.0, (self.h, self.w))
        else:
            frame = self.scene + self.rng.normal(0, 1.5, (self.h, self.w))
        return np.clip(np.rint(frame), 0, 255).astype(np.uint8)

    def block(self, mode: str = "E") -> bytes:
        if mode not in BLOCK_MODES:
            raise ValueError(f"mode must be one of {BLOCK_MODES}")
        import hashlib

        self.counter += 1
        if mode == "H":
            return os.urandom(32)
        h = hashlib.sha256(b"ESP32CAM-ENTROPY-v1" + self.raw_frame().tobytes())
        if mode == "E":
            h.update(os.urandom(32))
        h.update(struct.pack("<II", self.counter, time.monotonic_ns() & 0xFFFFFFFF))
        return h.digest()

    def info(self) -> str:
        return f"mock source | frame={self.w}x{self.h} gray | covered={self.covered}"

    def close(self):
        pass


def open_source(port: str | None, mock: bool = False, covered: bool = False):
    if mock:
        return MockCamSource(covered=covered)
    return CamSource(port or "/dev/ttyUSB0")
