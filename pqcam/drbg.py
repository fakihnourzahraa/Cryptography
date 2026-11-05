"""A SHAKE256-based random generator seeded from the ESP32-CAM.

Every random value the crypto code uses (ML-KEM key generation and
encapsulation, ML-DSA key generation and hedged signing, nonces, salts)
comes from CamDRBG.random_bytes().

Design (for the report):
  seed    = SHAKE256("CamDRBG-v1/seed" || camera blocks || os.urandom(32) || time)
  output  = SHAKE256("CamDRBG-v1/out"  || key || counter)   truncated to n bytes
  rekey   = key <- SHAKE256("CamDRBG-v1/next" || key)       after every request
  reseed  : fresh camera blocks are mixed in every `reseed_every` requests

Rekeying after each request gives forward secrecy: if the state leaks later,
earlier outputs can't be recomputed. Mixing in os.urandom means the generator
stays safe even if the camera were somehow predictable; pass mix_os=False to
run on camera entropy alone (useful to demonstrate the camera by itself).

This is a teaching construction. It follows the shape of NIST SP 800-90A
generators but is not a certified DRBG.
"""
from __future__ import annotations

import hashlib
import os
import struct
import time


def _shake(*parts: bytes, n: int) -> bytes:
    h = hashlib.shake_256()
    for p in parts:
        h.update(struct.pack("<I", len(p)))   # length-prefix so inputs can't run together
        h.update(p)
    return h.digest(n)


class CamDRBG:
    def __init__(self, source, seed_blocks: int = 8, mode: str = "E",
                 mix_os: bool = True, reseed_every: int = 64):
        self.source = source
        self.mode = mode
        self.mix_os = mix_os
        self.reseed_every = reseed_every
        self.requests = 0
        self.bytes_out = 0
        self.camera_blocks_used = 0
        self._key = _shake(b"CamDRBG-v1/seed", self._gather(seed_blocks), n=64)

    def _gather(self, blocks: int) -> bytes:
        material = b"".join(self.source.block(self.mode) for _ in range(blocks))
        self.camera_blocks_used += blocks
        if self.mix_os:
            material += os.urandom(32)
        material += struct.pack("<Q", time.time_ns())
        return material

    def reseed(self, blocks: int = 2) -> None:
        self._key = _shake(b"CamDRBG-v1/reseed", self._key, self._gather(blocks), n=64)

    def random_bytes(self, n: int) -> bytes:
        if n < 0:
            raise ValueError("n must be non-negative")
        if self.requests and self.requests % self.reseed_every == 0:
            self.reseed()
        out = _shake(b"CamDRBG-v1/out", self._key, struct.pack("<Q", self.requests), n=n)
        self._key = _shake(b"CamDRBG-v1/next", self._key, n=64)
        self.requests += 1
        self.bytes_out += n
        return out

    def stats(self) -> str:
        src = getattr(self.source, "name", "unknown")
        return (f"source={src} mode={self.mode} os_mix={self.mix_os} "
                f"camera_blocks={self.camera_blocks_used} requests={self.requests} bytes={self.bytes_out}")
