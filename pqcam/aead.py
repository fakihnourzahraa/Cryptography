"""Authenticated encryption from the SHA-3 family (no AES, no Ascon).

  keys       : enc_key, mac_key = KMAC256(shared_secret, context) with different customisation strings
  keystream  : SHAKE256(enc_key || nonce), squeezed as long as the data
  ciphertext : plaintext XOR keystream
  tag        : KMAC256(mac_key, header || ciphertext)       (encrypt-then-MAC)

The nonce must never repeat under the same key. Here every file gets a fresh
ML-KEM shared secret anyway, so keys are never reused, and the nonce adds a
second layer of protection.

Educational construction: built from standard primitives, but not a vetted
AEAD standard. Say so in the report.
"""
from __future__ import annotations

import hmac

from Crypto.Hash import KMAC256, SHAKE256

TAG_LEN = 32
CHUNK = 1 << 20  # 1 MiB


def derive_keys(secret: bytes, context: bytes) -> tuple[bytes, bytes]:
    """Split one secret (>= 32 bytes) into independent encryption and MAC keys."""
    def kmac(custom: bytes) -> bytes:
        h = KMAC256.new(key=secret, mac_len=32, custom=custom)
        h.update(context)
        return h.digest()
    return kmac(b"PQCF/enc-key"), kmac(b"PQCF/mac-key")


def _xor(a: bytes, b: bytes) -> bytes:
    return (int.from_bytes(a, "little") ^ int.from_bytes(b, "little")).to_bytes(len(a), "little")


class Sealer:
    """Streaming encrypt: feed plaintext chunks, get ciphertext chunks, then the tag."""

    def __init__(self, enc_key: bytes, mac_key: bytes, nonce: bytes, header: bytes):
        self._ks = SHAKE256.new(enc_key + nonce)
        self._mac = KMAC256.new(key=mac_key, mac_len=TAG_LEN, custom=b"PQCF/tag")
        self._mac.update(header)

    def update(self, plaintext: bytes) -> bytes:
        ct = _xor(plaintext, self._ks.read(len(plaintext)))
        self._mac.update(ct)
        return ct

    def tag(self) -> bytes:
        return self._mac.digest()


class Opener:
    """Streaming decrypt. Call verify() BEFORE trusting any output.

    For files, pqcf.py checks the tag over the whole ciphertext first and only
    then decrypts, so nothing unauthenticated is ever written out.
    """

    def __init__(self, enc_key: bytes, mac_key: bytes, nonce: bytes, header: bytes):
        self._ks = SHAKE256.new(enc_key + nonce)
        self._mac = KMAC256.new(key=mac_key, mac_len=TAG_LEN, custom=b"PQCF/tag")
        self._mac.update(header)

    def authenticate(self, ciphertext: bytes) -> None:
        self._mac.update(ciphertext)

    def verify(self, tag: bytes) -> bool:
        return hmac.compare_digest(self._mac.digest(), tag)

    def decrypt(self, ciphertext: bytes) -> bytes:
        return _xor(ciphertext, self._ks.read(len(ciphertext)))


def seal(enc_key, mac_key, nonce, header, plaintext: bytes) -> tuple[bytes, bytes]:
    s = Sealer(enc_key, mac_key, nonce, header)
    return s.update(plaintext), s.tag()


def open_(enc_key, mac_key, nonce, header, ciphertext: bytes, tag: bytes) -> bytes:
    o = Opener(enc_key, mac_key, nonce, header)
    o.authenticate(ciphertext)
    if not o.verify(tag):
        raise ValueError("authentication failed: wrong key or the data was modified")
    return o.decrypt(ciphertext)
