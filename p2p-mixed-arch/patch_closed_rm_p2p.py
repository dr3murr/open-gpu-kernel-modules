#!/usr/bin/env python3
"""Patch the exact NVIDIA R580.173.02 closed RM for PCIe BAR1 P2P.

This is deliberately a fail-closed, version-specific patcher.  Every symbol,
instruction byte, relocation type, relocation target, and input hash used by
the patch is checked before an output file is written.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import struct
from dataclasses import dataclass
from pathlib import Path

from elftools.elf.elffile import ELFFile
from elftools.elf.relocation import RelocationSection


EXPECTED_SHA256 = "232d1c78a39c5a8791b5739dc3e252ca724e384a334bbf47f9feb1c56e21eaae"


@dataclass(frozen=True)
class ExpectedSymbol:
    address: int
    size: int


EXPECTED_SYMBOLS = {
    # nvGpuOpsBuildExternalAllocPtes
    "_nv046768rm": ExpectedSymbol(0x0AF140, 0x12EF),
    # nvGpuOpsGetExternalAllocPtesOrPhysAddrs (the physical-address builder is
    # inlined in this function)
    "_nv046807rm": ExpectedSymbol(0x0B52A0, 0x11E2),
    # gpumgrGetGpuPhysFbAddr
    "_nv031415rm": ExpectedSymbol(0x03A9F0, 0x49),
    # gmmuFmtPtePhysAddrFld
    "_nv029452rm": ExpectedSymbol(0x5A4D20, 0x56),
    # KernelBus function-table initializer
    "_nv005902rm": ExpectedSymbol(0xB2D220, 0x2C74),
    # Pascal general peer mapping dispatchers and their later-generation
    # signature-compatible supersets.
    "_nv039345rm": ExpectedSymbol(0x15B630, 0x6B),
    "_nv039343rm": ExpectedSymbol(0x162870, 0xA6),
    "_nv039525rm": ExpectedSymbol(0x15BD00, 0x6B),
    "_nv039523rm": ExpectedSymbol(0x162920, 0xA6),
    # Default BAR1-P2P HAL stubs and GH100 implementations.
    "_nv039386rm": ExpectedSymbol(0xB2CEE0, 0x0A),
    "_nv039387rm": ExpectedSymbol(0x163810, 0x116),
    "_nv039334rm": ExpectedSymbol(0xB2CF20, 0x0A),
    "_nv039335rm": ExpectedSymbol(0x163960, 0x190),
    "_nv039514rm": ExpectedSymbol(0xB30DF0, 0x0A),
    "_nv039515rm": ExpectedSymbol(0x163AF0, 0x0B4),
    "_nv039449rm": ExpectedSymbol(0xB2CF30, 0x07),
    "_nv039448rm": ExpectedSymbol(0x163930, 0x02A),
    "_nv039474rm": ExpectedSymbol(0xB30DE0, 0x07),
    "_nv039473rm": ExpectedSymbol(0x1637A0, 0x06E),
    # Default static-BAR1 HAL stubs and the existing TU102 implementations.
    # Pascal has a sufficiently large BAR1 on this system, but its generated
    # table otherwise selects the unsupported stubs for these operations.
    "_nv039479rm": ExpectedSymbol(0xB2CEC0, 0x0A),
    "_nv039480rm": ExpectedSymbol(0x15E7D0, 0x1AE),
    "_nv039373rm": ExpectedSymbol(0xB2CED0, 0x0A),
    "_nv039374rm": ExpectedSymbol(0x15E980, 0x2C6),
    "_nv039372rm": ExpectedSymbol(0xB30EA0, 0x05),
    "_nv039371rm": ExpectedSymbol(0x15EC50, 0x0C2),
    "_nv039450rm": ExpectedSymbol(0xB2CEF0, 0x0A),
    "_nv039451rm": ExpectedSymbol(0x15ED20, 0x103),
    "_nv039346rm": ExpectedSymbol(0xB2CF00, 0x0A),
    "_nv039347rm": ExpectedSymbol(0x15EE30, 0x194),
    "_nv039440rm": ExpectedSymbol(0xB2CF10, 0x0A),
    "_nv039441rm": ExpectedSymbol(0x15EFD0, 0x48B),
}


class PatchError(RuntimeError):
    pass


def sha256(data: bytes | bytearray) -> str:
    return hashlib.sha256(data).hexdigest()


def rel32(opcode_address: int, target_address: int) -> bytes:
    displacement = target_address - (opcode_address + 5)
    if not -(1 << 31) <= displacement < (1 << 31):
        raise PatchError(
            f"rel32 from {opcode_address:#x} to {target_address:#x} is out of range"
        )
    return struct.pack("<i", displacement)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("/usr/src/nvidia-580.173.02/nvidia/nv-kernel.o_binary"),
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("work/stage/nv-kernel.o_binary.p2p"),
    )
    args = parser.parse_args()

    original = args.input.read_bytes()
    actual_hash = sha256(original)
    if actual_hash != EXPECTED_SHA256:
        raise PatchError(
            f"input SHA-256 mismatch: expected {EXPECTED_SHA256}, got {actual_hash}"
        )

    image = bytearray(original)
    elf = ELFFile(io.BytesIO(original))
    if elf["e_type"] != "ET_REL" or elf.get_machine_arch() != "x64":
        raise PatchError("input is not an x86-64 relocatable ELF object")

    text_section = elf.get_section_by_name(".text")
    symtab = elf.get_section_by_name(".symtab")
    rela_text = elf.get_section_by_name(".rela.text")
    if text_section is None or symtab is None or not isinstance(rela_text, RelocationSection):
        raise PatchError("required .text, .symtab, or .rela.text section is absent")
    if int(text_section["sh_addr"]) != 0:
        raise PatchError("unexpected nonzero .text section address")

    text_file_offset = int(text_section["sh_offset"])
    text_size = int(text_section["sh_size"])
    text_section_index = elf.get_section_index(".text")

    symbols_by_name: dict[str, list[tuple[int, object]]] = {}
    for index, symbol in enumerate(symtab.iter_symbols()):
        symbols_by_name.setdefault(symbol.name, []).append((index, symbol))

    def unique_symbol(name: str) -> tuple[int, object]:
        matches = symbols_by_name.get(name, [])
        if len(matches) != 1:
            raise PatchError(f"expected one symbol named {name}, found {len(matches)}")
        return matches[0]

    for name, expected in EXPECTED_SYMBOLS.items():
        _, symbol = unique_symbol(name)
        address = int(symbol["st_value"])
        size = int(symbol["st_size"])
        section_index = symbol["st_shndx"]
        if address != expected.address or size != expected.size:
            raise PatchError(
                f"symbol {name} mismatch: expected {expected.address:#x}/{expected.size:#x}, "
                f"got {address:#x}/{size:#x}"
            )
        if section_index != text_section_index:
            raise PatchError(f"symbol {name} is not in .text")

    manifest: list[str] = []

    def patch_text(address: int, expected: bytes, replacement: bytes, label: str) -> None:
        if len(expected) != len(replacement):
            raise PatchError(f"{label}: replacement changes the byte count")
        if address < 0 or address + len(expected) > text_size:
            raise PatchError(f"{label}: .text range is invalid")
        file_offset = text_file_offset + address
        actual = bytes(image[file_offset : file_offset + len(expected)])
        if actual != expected:
            raise PatchError(
                f"{label}: bytes at .text+{address:#x} mismatch; "
                f"expected {expected.hex()}, got {actual.hex()}"
            )
        image[file_offset : file_offset + len(expected)] = replacement
        manifest.append(
            f"text {address:#010x} {label}: {expected.hex()} -> {replacement.hex()}"
        )

    # Index all .rela.text entries and retain the exact on-disk entry offset.
    relocations: dict[int, tuple[int, object, object]] = {}
    linked_symtab = elf.get_section(int(rela_text["sh_link"]))
    entry_size = int(rela_text["sh_entsize"])
    for index, relocation in enumerate(rela_text.iter_relocations()):
        offset = int(relocation["r_offset"])
        if offset in relocations:
            raise PatchError(f"duplicate .rela.text relocation at {offset:#x}")
        symbol = linked_symtab.get_symbol(int(relocation["r_info_sym"]))
        entry_offset = int(rela_text["sh_offset"]) + index * entry_size
        relocations[offset] = (entry_offset, relocation, symbol)

    def rewrite_relocation(
        offset: int,
        *,
        expected_type: int,
        expected_symbol_name: str,
        expected_symbol_index: int | None = None,
        expected_addend: int,
        target_symbol_name: str | None,
        target_type: int | None = None,
        target_addend: int = 0,
        label: str,
    ) -> None:
        try:
            entry_offset, relocation, old_symbol = relocations[offset]
        except KeyError as exc:
            raise PatchError(f"{label}: no relocation at .text+{offset:#x}") from exc

        old_type = int(relocation["r_info_type"])
        old_symbol_index = int(relocation["r_info_sym"])
        old_addend = int(relocation["r_addend"])
        if (
            old_type != expected_type
            or old_symbol.name != expected_symbol_name
            or old_addend != expected_addend
            or (
                expected_symbol_index is not None
                and old_symbol_index != expected_symbol_index
            )
        ):
            raise PatchError(
                f"{label}: relocation mismatch at {offset:#x}: "
                f"type={old_type}, sym={old_symbol_index}/{old_symbol.name!r}, "
                f"addend={old_addend:+#x}"
            )

        if target_symbol_name is None:
            new_symbol_index = 0
            new_type = 0 if target_type is None else target_type
        else:
            new_symbol_index, _ = unique_symbol(target_symbol_name)
            new_type = old_type if target_type is None else target_type
        new_info = (new_symbol_index << 32) | new_type
        struct.pack_into("<QQq", image, entry_offset, offset, new_info, target_addend)
        manifest.append(
            f"rela {offset:#010x} {label}: "
            f"type{old_type} {old_symbol.name or '<.text>'}{old_addend:+#x} -> "
            f"type{new_type} {target_symbol_name or '<none>'}{target_addend:+#x}"
        )

    # Make the Pascal general P2P dispatcher accept the BAR1/PCIe mapping type.
    rewrite_relocation(
        0xB2F1B6,
        expected_type=11,
        expected_symbol_name="_nv039345rm",
        expected_addend=0,
        target_symbol_name="_nv039343rm",
        label="Pascal create-P2P dispatcher -> BAR1-capable superset",
    )
    rewrite_relocation(
        0xB2EB4C,
        expected_type=11,
        expected_symbol_name="_nv039525rm",
        expected_addend=0,
        target_symbol_name="_nv039523rm",
        label="Pascal remove-P2P dispatcher -> BAR1-capable superset",
    )

    # Redirect every default BAR1-P2P KernelBus HAL assignment in the closed
    # initializer to its existing GH100 implementation.  The old targets are
    # .text section-symbol relocations, hence symbol index 1 plus an addend.
    default_hal_relocations = {
        # get DMA info
        0xB2E14D: (0xB2CEE0, "_nv039387rm", "get BAR1-P2P DMA info"),
        0xB2F73F: (0xB2CEE0, "_nv039387rm", "get BAR1-P2P DMA info"),
        0xB2FBD3: (0xB2CEE0, "_nv039387rm", "get BAR1-P2P DMA info"),
        # create mapping
        0xB2DC1D: (0xB2CF20, "_nv039335rm", "create BAR1-P2P mapping"),
        0xB2E179: (0xB2CF20, "_nv039335rm", "create BAR1-P2P mapping"),
        0xB2FC03: (0xB2CF20, "_nv039335rm", "create BAR1-P2P mapping"),
        # remove mapping
        0xB2DC3B: (0xB30DF0, "_nv039515rm", "remove BAR1-P2P mapping"),
        0xB2E184: (0xB30DF0, "_nv039515rm", "remove BAR1-P2P mapping"),
        0xB2FC0E: (0xB30DF0, "_nv039515rm", "remove BAR1-P2P mapping"),
        # mapping exists
        0xB2DC46: (0xB2CF30, "_nv039448rm", "query BAR1-P2P mapping"),
        0xB2E18F: (0xB2CF30, "_nv039448rm", "query BAR1-P2P mapping"),
        0xB2FC19: (0xB2CF30, "_nv039448rm", "query BAR1-P2P mapping"),
        # supported
        0xB2DC55: (0xB30DE0, "_nv039473rm", "BAR1-P2P support predicate"),
        0xB2E19A: (0xB30DE0, "_nv039473rm", "BAR1-P2P support predicate"),
        0xB2FC24: (0xB30DE0, "_nv039473rm", "BAR1-P2P support predicate"),
    }
    for offset, (old_addend, target, description) in default_hal_relocations.items():
        rewrite_relocation(
            offset,
            expected_type=11,
            expected_symbol_name="",
            expected_symbol_index=1,
            expected_addend=old_addend,
            target_symbol_name=target,
            label=f"default HAL: {description}",
        )

    # Static BAR1 is already supported by the common/TU102 implementation,
    # but pre-Turing chips are wired to fail-closed stubs.  Redirect every
    # default assignment in this exact initializer so the P100 can construct
    # the full-FB BAR1 descriptor required by the GH100 BAR1-P2P transport.
    static_bar1_hal_relocations = {
        # support predicate
        0xB2DB65: (0xB2CEC0, "_nv039480rm", "static BAR1 support predicate"),
        0xB2E118: (0xB2CEC0, "_nv039480rm", "static BAR1 support predicate"),
        0xB2FB97: (0xB2CEC0, "_nv039480rm", "static BAR1 support predicate"),
        # enable mapping
        0xB2E12D: (0xB2CED0, "_nv039374rm", "enable static BAR1 mapping"),
        0xB2FA77: (0xB2CED0, "_nv039374rm", "enable static BAR1 mapping"),
        0xB2FBA2: (0xB2CED0, "_nv039374rm", "enable static BAR1 mapping"),
        # disable mapping
        0xB2E142: (0xB30EA0, "_nv039371rm", "disable static BAR1 mapping"),
        0xB2FA95: (0xB30EA0, "_nv039371rm", "disable static BAR1 mapping"),
        0xB2FBC8: (0xB30EA0, "_nv039371rm", "disable static BAR1 mapping"),
        # acquire mapping
        0xB2DBCF: (0xB2CEF0, "_nv039451rm", "increase static BAR1 refcount"),
        0xB2E158: (0xB2CEF0, "_nv039451rm", "increase static BAR1 refcount"),
        0xB2FAB5: (0xB2CEF0, "_nv039451rm", "increase static BAR1 refcount"),
        0xB2FBE2: (0xB2CEF0, "_nv039451rm", "increase static BAR1 refcount"),
        # release mapping
        0xB2DBF5: (0xB2CF00, "_nv039347rm", "decrease static BAR1 refcount"),
        0xB2E163: (0xB2CF00, "_nv039347rm", "decrease static BAR1 refcount"),
        0xB2FBED: (0xB2CF00, "_nv039347rm", "decrease static BAR1 refcount"),
        # resolve an FB allocation through the static aperture
        0xB2DC00: (0xB2CF10, "_nv039441rm", "get static BAR1 FB aperture"),
        0xB2E16E: (0xB2CF10, "_nv039441rm", "get static BAR1 FB aperture"),
        0xB2FBF8: (0xB2CF10, "_nv039441rm", "get static BAR1 FB aperture"),
    }
    for offset, (old_addend, target, description) in static_bar1_hal_relocations.items():
        rewrite_relocation(
            offset,
            expected_type=11,
            expected_symbol_name="",
            expected_symbol_index=1,
            expected_addend=old_addend,
            target_symbol_name=target,
            label=f"default HAL: {description}",
        )

    # gmmuFmtPtePhysAddrFld: PEER must use the sysmem physical-address field.
    patch_text(
        0x5A4D2B,
        b"\xb0",
        b"\x30",
        "PEER PTE address field -> sysmem field",
    )

    # The PTE builder has two aperture setters: legacy formats and GMMU v3.
    # Detour each five-byte table lookup to a peer-only index adjustment in the
    # now-unused legacy peer/fabric block.  rdx is aperture*2 here: peer is 4
    # and SYS_NONCOH is 6.  Mixed-generation BAR1 mappings must remain
    # non-coherent; in particular, Blackwell must not encode a Pascal peer as
    # SYS_COH.  This matches the mixed-generation handling added by the newer
    # 610.57.04-p2p-v2 patch series.
    patch_text(
        0xAF8B4,
        bytes.fromhex("0f b6 44 10 01"),
        b"\xe9" + rel32(0xAF8B4, 0xB0041),
        "legacy aperture setter detour",
    )
    patch_text(
        0xAFF51,
        bytes.fromhex("0f b6 44 10 01"),
        b"\xe9" + rel32(0xAFF51, 0xB0055),
        "GMMU-v3 aperture setter detour",
    )

    # Replace the PTE builder's old peer/NVLink entry with:
    #   pRemoteGpu = pMemDesc->pGpu; gpumgrGetGpuPhysFbAddr(pRemoteGpu);
    #   fabricBaseAddress = result; goto common;
    old_peer_entry = bytes.fromhex(
        "48 8b 7d 10 48 c7 c6 00 00 00 00 e8 00 00 00 00 "
        "48 8b 4d 18 8b 74 24 50 48"
    )
    new_peer_entry = (
        bytes.fromhex("48 8b 3b")
        + bytes.fromhex("0f 1f 84 00 00 00 00 00")
        + b"\xe8\x00\x00\x00\x00"
        + bytes.fromhex("48 89 45 10")
        + b"\xe9"
        + rel32(0xB003C, 0xAF932)
    )
    patch_text(
        0xB0028,
        old_peer_entry,
        new_peer_entry,
        "PTE peer fabric base -> remote GPU physical BAR1 base",
    )
    rewrite_relocation(
        0xB002F,
        expected_type=11,
        expected_symbol_name="_nv002939rm",
        expected_addend=0,
        target_symbol_name=None,
        label="remove overwritten PTE peer class relocation",
    )
    rewrite_relocation(
        0xB0034,
        expected_type=2,
        expected_symbol_name="_nv005245rm",
        expected_addend=-4,
        target_symbol_name="_nv031415rm",
        target_addend=-4,
        label="PTE peer helper call -> gpumgrGetGpuPhysFbAddr",
    )

    legacy_aperture_helper = (
        bytes.fromhex("48 83 fa 04 75 04 48 83 c2 02 0f b6 44 10 01")
        + b"\xe9"
        + rel32(0xB0050, 0xAF8B9)
    )
    v3_aperture_helper = (
        bytes.fromhex("48 83 fa 04 75 04 48 83 c2 02 0f b6 44 10 01")
        + b"\xe9"
        + rel32(0xB0064, 0xAFF56)
    )
    patch_text(
        0xB0041,
        bytes.fromhex(
            "89 df 48 89 45 10 0f b7 91 fa 00 00 00 4c 01 fa 8b 02 23 81 "
            "f4 00 00 00 0f b7 89 f8 00 00 00 d3 e6 09 f0 89 02 e8 00 00"
        ),
        legacy_aperture_helper + v3_aperture_helper,
        "peer-only legacy/v3 aperture encoding helpers",
    )
    rewrite_relocation(
        0xB0067,
        expected_type=2,
        expected_symbol_name="_nv043696rm",
        expected_addend=-4,
        target_symbol_name=None,
        label="remove overwritten aperture-cave relocation",
    )

    # The physical-address builder is inlined in _nv046807rm.  Its peer block
    # uses r11=pMemDesc and rbx=fabricBaseAddress.  A same-.text raw rel32 call
    # is safe in an ET_REL object because all addresses move together when this
    # input section is linked into nvidia.ko.
    physical_peer_patch = (
        bytes.fromhex("49 8b 3b")
        + b"\xe8"
        + rel32(0xB6061, 0x03A9F0)
        + bytes.fromhex("48 89 c3")
        + b"\xe9"
        + rel32(0xB6069, 0xB5706)
    )
    patch_text(
        0xB605E,
        bytes.fromhex(
            "48 8b 7d 58 48 c7 c6 00 00 00 00 4c 89 55 48 4c"
        ),
        physical_peer_patch,
        "physical-address peer fabric base -> remote GPU physical BAR1 base",
    )
    rewrite_relocation(
        0xB6065,
        expected_type=11,
        expected_symbol_name="_nv002939rm",
        expected_addend=0,
        target_symbol_name=None,
        label="remove overwritten physical-address peer class relocation",
    )

    if len(image) != len(original):
        raise PatchError("internal error: output size changed")
    if image == original:
        raise PatchError("internal error: patch made no changes")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(image)
    print(f"input:  {args.input}")
    print(f"sha256: {actual_hash}")
    print(f"output: {args.output}")
    print(f"sha256: {sha256(image)}")
    print(f"size:   {len(image)} bytes (unchanged)")
    print(f"changes: {len(manifest)}")
    for entry in manifest:
        print(f"  {entry}")


if __name__ == "__main__":
    main()
