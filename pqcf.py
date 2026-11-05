#!/usr/bin/env python3
"""pqcf — post-quantum file encryption with ESP32-CAM entropy.

  python pqcf.py info                         # check the camera connection
  python pqcf.py keygen                       # make your ML-KEM + ML-DSA key pair
  python pqcf.py encrypt notes.txt            # -> notes.txt.pqcf
  python pqcf.py decrypt notes.txt.pqcf       # -> notes.txt (verifies first)

Common options: --port /dev/ttyUSB0, --mock (no hardware), --camera-only
(don't mix in os.urandom). The key password can come from the PQCF_PASSWORD
environment variable for scripted runs.
"""
from __future__ import annotations

import argparse
import getpass
import os
import sys
import time

from pqcam.drbg import CamDRBG
from pqcam.source import DeviceError, open_source
from pqcam import pqfile

DEFAULT_KEYDIR = "keys"


def make_drbg(args):
    src = open_source(args.port, mock=args.mock)
    if args.mock:
        print("!! MOCK entropy source: fine for testing, NOT camera entropy.", file=sys.stderr)
    return src, CamDRBG(src, mode="C" if args.camera_only else "E", mix_os=not args.camera_only)


def get_password(confirm: bool) -> str | None:
    if "PQCF_PASSWORD" in os.environ:
        return os.environ["PQCF_PASSWORD"] or None
    pw = getpass.getpass("Private key password (empty = no password): ")
    if pw and confirm and getpass.getpass("Repeat password: ") != pw:
        sys.exit("passwords did not match")
    return pw or None


def load_keys(keydir: str) -> pqfile.PrivateKey:
    path = os.path.join(keydir, "private.key")
    if not os.path.exists(path):
        sys.exit(f"no key at {path}: run 'python pqcf.py keygen' first")
    with open(path, "rb") as f:
        data = f.read()
    pw = get_password(confirm=False) if data[7] else None
    return pqfile.PrivateKey.from_bytes(data, pw)


def cmd_info(args):
    src = open_source(args.port, mock=args.mock)
    print(src.info())
    for mode in ("E", "C", "H"):
        print(f"block {mode}: {src.block(mode).hex()}")
    frame = src.raw_frame()
    print(f"raw frame: {frame.shape[1]}x{frame.shape[0]}, mean={frame.mean():.1f}, std={frame.std():.1f}")
    src.close()


def cmd_keygen(args):
    os.makedirs(args.keys, exist_ok=True)
    priv_path = os.path.join(args.keys, "private.key")
    pub_path = os.path.join(args.keys, "public.key")
    if os.path.exists(priv_path) and not args.force:
        sys.exit(f"{priv_path} already exists (use --force to overwrite)")
    src, drbg = make_drbg(args)
    t = time.perf_counter()
    priv = pqfile.generate_keys(drbg, args.level)
    dt = time.perf_counter() - t
    pw = get_password(confirm=True)
    with open(priv_path, "wb") as f:
        f.write(priv.to_bytes(pw, drbg))
    os.chmod(priv_path, 0o600)
    with open(pub_path, "wb") as f:
        f.write(priv.public.to_bytes())
    src.close()
    kem, sig = pqfile.KEMS[priv.kem_id][0], pqfile.SIGS[priv.sig_id][0]
    print(f"{kem} + {sig} keys written to {args.keys}/ in {dt:.2f}s "
          f"({'password-protected' if pw else 'NOT password-protected'})")
    print(f"key id {priv.public.key_id.hex()} | {drbg.stats()}")


def cmd_encrypt(args):
    priv = load_keys(args.keys)
    src, drbg = make_drbg(args)
    out = args.output or args.input + ".pqcf"
    t = time.perf_counter()
    r = pqfile.encrypt_file(args.input, out, priv, priv.public, drbg)
    src.close()
    print(f"{args.input} -> {out}  [{r['kem']} + {r['sig']}, +{r['overhead_bytes']} bytes, "
          f"{time.perf_counter() - t:.2f}s]")
    print(drbg.stats())


def cmd_decrypt(args):
    priv = load_keys(args.keys)
    out = args.output or (args.input[:-5] if args.input.endswith(".pqcf") else args.input + ".dec")
    if os.path.exists(out) and not args.force:
        sys.exit(f"{out} already exists (use --force or -o)")
    t = time.perf_counter()
    r = pqfile.decrypt_file(args.input, out, priv)
    print(f"signature OK, tag OK: {args.input} -> {out}  [{r['kem']} + {r['sig']}, "
          f"{time.perf_counter() - t:.2f}s]")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default="/dev/ttyUSB0", help="ESP32-CAM serial port")
    p.add_argument("--mock", action="store_true", help="use the mock source (no hardware)")
    p.add_argument("--camera-only", action="store_true",
                   help="seed from camera frames only (no hardware RNG, no os.urandom)")
    p.add_argument("--keys", default=DEFAULT_KEYDIR, help="key directory (default: keys/)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="test the entropy source")
    k = sub.add_parser("keygen", help="generate ML-KEM + ML-DSA keys")
    k.add_argument("--level", choices=["1", "3", "5"], default="3",
                   help="NIST level: 1 = ML-KEM-512/ML-DSA-44, 3 = 768/65 (default), 5 = 1024/87")
    k.add_argument("--force", action="store_true")
    e = sub.add_parser("encrypt", help="encrypt and sign a file")
    e.add_argument("input"); e.add_argument("-o", "--output")
    d = sub.add_parser("decrypt", help="verify and decrypt a file")
    d.add_argument("input"); d.add_argument("-o", "--output"); d.add_argument("--force", action="store_true")

    args = p.parse_args(argv)
    try:
        {"info": cmd_info, "keygen": cmd_keygen, "encrypt": cmd_encrypt, "decrypt": cmd_decrypt}[args.cmd](args)
    except (DeviceError, ValueError, OSError) as exc:
        sys.exit(f"error: {exc}")


if __name__ == "__main__":
    main()
