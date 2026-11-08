#!/usr/bin/env python3
"""Create a validated R580 libcuda with mixed-arch P2P fully advertised.

The input must be the untouched 580.173.02 library saved before installation.
All three edits are exact-byte and exact-hash checked so a driver upgrade fails
closed instead of applying offsets to an unknown binary.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil


SOURCE = Path("work/original/libcuda.so.580.173.02.stock")
OUTPUT = Path("work/stage/libcuda.so.580.173.02.p2p")
SOURCE_SHA256 = "266916dac6c5e7e4655526570cf303a6cc017880d2552960abc66017a7c98cf4"

PATCHES = (
    # libcuda internal cuDeviceCanAccessPeer predicate at 0x2c2230:
    #   0x2c2273: cmp rax, rdx       (compare architecture codes)
    #   0x2c2276: je  0x2c22bc       (continue only when equal)
    # Changing JE to JMP skips only the mixed-architecture rejection.
    (
        "mixed-architecture peer predicate",
        0x2C2276,
        bytes.fromhex("74 44"),
        bytes.fromhex("eb 44"),
    ),
    # cuDeviceGetP2PAttribute has a six-way jump table. Attribute 0x02
    # (ACCESS_SUPPORTED) used a stale cached byte at record+0x11, while
    # cuDeviceCanAccessPeer used the now-patched live predicate above. Reuse
    # six bytes of compiler padding immediately before that case to call the
    # live predicate with the two already-preserved GPU object pointers. This
    # affects access support only; native atomics and array access keep their
    # original cases.
    (
        "ACCESS_SUPPORTED live-predicate case",
        0x1AB16A,
        bytes.fromhex(
            "66 0f 1f 44 00 00 "
            "0f b6 95 61 fd ff ff 89 13 eb b4 "
            "0f 1f 44 00 00"
        ),
        bytes.fromhex(
            "4c 89 ef "          # mov r13, rdi (source GPU object)
            "4c 89 f6 "          # mov r14, rsi (destination GPU object)
            "e8 bb 70 11 00 "    # call 0x2c2230 (live peer predicate)
            "0f b6 c0 "          # movzx al, eax
            "89 03 "             # mov eax, [rbx] (attribute result)
            "31 c0 "             # CUDA_SUCCESS
            "eb b1 "             # jump common epilogue at 0x1ab12f
            "66 90"              # preserve alignment
        ),
    ),
    # Redirect only jump-table entry 2 from the old case at 0x1ab170 to the
    # expanded case at 0x1ab16a. The table base is 0x117e19c.
    (
        "ACCESS_SUPPORTED jump-table entry",
        0x117E1A4,
        bytes.fromhex("d4 cf 02 ff"),  # 0x1ab170 - 0x117e19c
        bytes.fromhex("ce cf 02 ff"),  # 0x1ab16a - 0x117e19c
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    actual_sha256 = sha256(args.source)
    if actual_sha256 != SOURCE_SHA256:
        raise SystemExit(
            f"Refusing unknown libcuda: expected {SOURCE_SHA256}, got {actual_sha256}"
        )

    data = bytearray(args.source.read_bytes())
    for name, offset, expected, replacement in PATCHES:
        if len(expected) != len(replacement):
            raise SystemExit(f"Internal error: size-changing patch {name!r}")
        actual = bytes(data[offset : offset + len(expected)])
        if actual != expected:
            raise SystemExit(
                f"Unexpected bytes for {name} at {offset:#x}: "
                f"expected {expected.hex(' ')}, got {actual.hex(' ')}"
            )
        data[offset : offset + len(replacement)] = replacement

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    shutil.copymode(args.source, args.output)

    print(f"source_sha256={actual_sha256}")
    for name, offset, expected, replacement in PATCHES:
        print(f"patch={name} offset={offset:#x}")
        print(f"  {expected.hex(' ')} -> {replacement.hex(' ')}")
    print(f"output={args.output}")
    print(f"output_sha256={sha256(args.output)}")


if __name__ == "__main__":
    main()
