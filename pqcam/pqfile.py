"""ML-KEM + ML-DSA file encryption, with all randomness from the camera DRBG.

Uses kyber-py and dilithium-py: readable pure-Python implementations of
FIPS 203 (ML-KEM) and FIPS 204 (ML-DSA). They let us plug our own random
source in directly, which is the point of the project. They are slow and not
side-channel hardened (fine for a laptop demo; the benchmarks use liboqs).

File layout (.pqcf):
    header : "PQCF" | ver | kem_id | sig_id | 0 | key_id(8) | kem_ct | nonce(16) | pt_len(u64)
    body   : ciphertext (pt_len bytes)
    tag    : KMAC256 tag (32)
    sig    : ML-DSA signature over SHA3-512(header || ciphertext || tag)

Decryption checks the signature and the tag over the WHOLE file before
writing any plaintext.
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import struct
import tempfile

from kyber_py.ml_kem import ML_KEM_512, ML_KEM_768, ML_KEM_1024
from dilithium_py.ml_dsa import ML_DSA_44, ML_DSA_65, ML_DSA_87

from . import aead

# id -> (name, implementation, ciphertext or signature size)
KEMS = {1: ("ML-KEM-512", ML_KEM_512, 768), 2: ("ML-KEM-768", ML_KEM_768, 1088), 3: ("ML-KEM-1024", ML_KEM_1024, 1568)}
SIGS = {1: ("ML-DSA-44", ML_DSA_44, 2420), 2: ("ML-DSA-65", ML_DSA_65, 3309), 3: ("ML-DSA-87", ML_DSA_87, 4627)}
LEVELS = {"1": (1, 1), "3": (2, 2), "5": (3, 3)}   # NIST security level -> (kem_id, sig_id)

FILE_MAGIC, PUB_MAGIC, PRIV_MAGIC = b"PQCF", b"PQCP", b"PQCK"
VERSION = 1
NONCE_LEN = 16
SIG_CTX = b"PQCF-v1"


@contextlib.contextmanager
def using_rng(impl, drbg):
    """Temporarily route a kyber-py / dilithium-py object's randomness to our DRBG."""
    old = impl.random_bytes
    impl.random_bytes = drbg.random_bytes
    try:
        yield impl
    finally:
        impl.random_bytes = old


def _key_id(ek: bytes, sig_pk: bytes) -> bytes:
    return hashlib.sha3_256(ek + sig_pk).digest()[:8]


# ------------------------------------------------------------------ keys
class PublicKey:
    def __init__(self, kem_id, sig_id, ek, sig_pk):
        self.kem_id, self.sig_id, self.ek, self.sig_pk = kem_id, sig_id, ek, sig_pk
        self.key_id = _key_id(ek, sig_pk)

    def to_bytes(self) -> bytes:
        return (PUB_MAGIC + bytes([VERSION, self.kem_id, self.sig_id, 0])
                + struct.pack("<I", len(self.ek)) + self.ek
                + struct.pack("<I", len(self.sig_pk)) + self.sig_pk)

    @classmethod
    def from_bytes(cls, b: bytes) -> "PublicKey":
        if b[:4] != PUB_MAGIC:
            raise ValueError("not a PQCF public key file")
        kem_id, sig_id = b[5], b[6]
        off = 8
        (n,) = struct.unpack_from("<I", b, off); off += 4
        ek = b[off:off + n]; off += n
        (m,) = struct.unpack_from("<I", b, off); off += 4
        return cls(kem_id, sig_id, ek, b[off:off + m])


class PrivateKey:
    def __init__(self, kem_id, sig_id, dk, sig_sk, public: PublicKey):
        self.kem_id, self.sig_id, self.dk, self.sig_sk, self.public = kem_id, sig_id, dk, sig_sk, public

    def _inner(self) -> bytes:
        pub = self.public.to_bytes()
        return b"".join(struct.pack("<I", len(x)) + x for x in (self.dk, self.sig_sk, pub))

    def to_bytes(self, password: str | None, drbg) -> bytes:
        head = PRIV_MAGIC + bytes([VERSION, self.kem_id, self.sig_id])
        inner = self._inner()
        if not password:
            return head + b"\x00" + inner
        salt = drbg.random_bytes(16)
        nonce = drbg.random_bytes(NONCE_LEN)
        k = hashlib.scrypt(password.encode(), salt=salt, n=2**15, r=8, p=1, maxmem=2**26, dklen=64)
        enc_key, mac_key = aead.derive_keys(k, b"PQCK/private-key")
        header = head + b"\x01" + salt + nonce
        ct, tag = aead.seal(enc_key, mac_key, nonce, header, inner)
        return header + ct + tag

    @classmethod
    def from_bytes(cls, b: bytes, password: str | None) -> "PrivateKey":
        if b[:4] != PRIV_MAGIC:
            raise ValueError("not a PQCF private key file")
        kem_id, sig_id, encrypted = b[5], b[6], b[7]
        if encrypted:
            if not password:
                raise ValueError("this private key is password-protected")
            salt, nonce = b[8:24], b[24:40]
            header = b[:40]
            ct, tag = b[40:-aead.TAG_LEN], b[-aead.TAG_LEN:]
            k = hashlib.scrypt(password.encode(), salt=salt, n=2**15, r=8, p=1, maxmem=2**26, dklen=64)
            enc_key, mac_key = aead.derive_keys(k, b"PQCK/private-key")
            try:
                inner = aead.open_(enc_key, mac_key, nonce, header, ct, tag)
            except ValueError:
                raise ValueError("wrong password (or the key file is damaged)") from None
        else:
            inner = b[8:]
        parts, off = [], 0
        for _ in range(3):
            (n,) = struct.unpack_from("<I", inner, off); off += 4
            parts.append(inner[off:off + n]); off += n
        dk, sig_sk, pub = parts
        return cls(kem_id, sig_id, dk, sig_sk, PublicKey.from_bytes(pub))


def generate_keys(drbg, level: str = "3") -> PrivateKey:
    kem_id, sig_id = LEVELS[level]
    with using_rng(KEMS[kem_id][1], drbg) as kem:
        ek, dk = kem.keygen()
    with using_rng(SIGS[sig_id][1], drbg) as sig:
        sig_pk, sig_sk = sig.keygen()
    pub = PublicKey(kem_id, sig_id, ek, sig_pk)
    return PrivateKey(kem_id, sig_id, dk, sig_sk, pub)


# ------------------------------------------------------------------ files
def _header(pub: PublicKey, kem_ct: bytes, nonce: bytes, pt_len: int) -> bytes:
    return (FILE_MAGIC + bytes([VERSION, pub.kem_id, pub.sig_id, 0]) + pub.key_id
            + kem_ct + nonce + struct.pack("<Q", pt_len))


def _header_len(kem_id: int) -> int:
    return 8 + 8 + KEMS[kem_id][2] + NONCE_LEN + 8


def encrypt_file(src: str, dst: str, priv: PrivateKey, recipient: PublicKey, drbg) -> dict:
    """Encrypt to `recipient` (your own public key) and sign with `priv`'s ML-DSA key."""
    kem_name, kem, _ = KEMS[recipient.kem_id]
    sig_name, sig, sig_len = SIGS[priv.sig_id]
    pt_len = os.path.getsize(src)

    with using_rng(kem, drbg):
        shared_secret, kem_ct = kem.encaps(recipient.ek)
    nonce = drbg.random_bytes(NONCE_LEN)
    header = _header(recipient, kem_ct, nonce, pt_len)
    enc_key, mac_key = aead.derive_keys(shared_secret, header[:16])
    sealer = aead.Sealer(enc_key, mac_key, nonce, header)
    digest = hashlib.sha3_512(b"PQCF/sig")
    digest.update(header)

    with open(src, "rb") as fin, open(dst, "wb") as fout:
        fout.write(header)
        while chunk := fin.read(aead.CHUNK):
            ct = sealer.update(chunk)
            digest.update(ct)
            fout.write(ct)
        tag = sealer.tag()
        digest.update(tag)
        fout.write(tag)
        with using_rng(sig, drbg):            # hedged signing: fresh camera randomness
            signature = sig.sign(priv.sig_sk, digest.digest(), ctx=SIG_CTX)
        assert len(signature) == sig_len
        fout.write(signature)
    return {"kem": kem_name, "sig": sig_name, "plaintext_bytes": pt_len,
            "output_bytes": os.path.getsize(dst), "overhead_bytes": os.path.getsize(dst) - pt_len}


def decrypt_file(src: str, dst: str, priv: PrivateKey, sender: PublicKey | None = None) -> dict:
    """Verify signature and tag over the whole file, then decrypt.

    `sender` is whose signature to expect; defaults to your own key.
    """
    sender = sender or priv.public
    total = os.path.getsize(src)
    with open(src, "rb") as f:
        fixed = f.read(8 + 8)
        if fixed[:4] != FILE_MAGIC:
            raise ValueError("not a .pqcf file")
        kem_id, sig_id = fixed[5], fixed[6]
        key_id = fixed[8:16]
        if key_id != priv.public.key_id:
            raise ValueError("this file was encrypted for a different key")
        kem_name, kem, ct_len = KEMS[kem_id]
        sig_name, sig, sig_len = SIGS[sig_id]
        rest = f.read(ct_len + NONCE_LEN + 8)
        header = fixed + rest
        kem_ct = rest[:ct_len]
        nonce = rest[ct_len:ct_len + NONCE_LEN]
        (pt_len,) = struct.unpack("<Q", rest[-8:])
        if total != _header_len(kem_id) + pt_len + aead.TAG_LEN + sig_len:
            raise ValueError("file is truncated or has extra data")

        shared_secret = kem.decaps(priv.dk, kem_ct)
        enc_key, mac_key = aead.derive_keys(shared_secret, header[:16])

        # Pass 1: hash for the signature and MAC the ciphertext. No output yet.
        opener = aead.Opener(enc_key, mac_key, nonce, header)
        digest = hashlib.sha3_512(b"PQCF/sig")
        digest.update(header)
        body_start = f.tell()
        left = pt_len
        while left:
            chunk = f.read(min(aead.CHUNK, left))
            left -= len(chunk)
            opener.authenticate(chunk)
            digest.update(chunk)
        tag = f.read(aead.TAG_LEN)
        digest.update(tag)
        signature = f.read(sig_len)

        if not sig.verify(sender.sig_pk, digest.digest(), signature, ctx=SIG_CTX):
            raise ValueError("signature check FAILED: the file was modified or not signed by this key")
        if not opener.verify(tag):
            raise ValueError("authentication tag FAILED: the file was modified")

        # Pass 2: decrypt to a temp file, then move it into place.
        f.seek(body_start)
        out_dir = os.path.dirname(os.path.abspath(dst))
        fd, tmp = tempfile.mkstemp(dir=out_dir, prefix=".pqcf-")
        try:
            with os.fdopen(fd, "wb") as fout:
                left = pt_len
                while left:
                    chunk = f.read(min(aead.CHUNK, left))
                    left -= len(chunk)
                    fout.write(opener.decrypt(chunk))
            os.replace(tmp, dst)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)
            raise
    return {"kem": kem_name, "sig": sig_name, "plaintext_bytes": pt_len}
