#!/usr/bin/env python3
"""Locate source-equivalent routines in NVIDIA's obfuscated relocatable RM ELF."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from capstone import CS_ARCH_X86, CS_MODE_64, Cs
from capstone.x86 import (
    X86_OP_IMM,
    X86_OP_MEM,
    X86_REG_RAX,
    X86_REG_RBP,
    X86_REG_RDI,
    X86_REG_RSI,
)
from elftools.elf.elffile import ELFFile


CLOSED = Path("/usr/src/nvidia-580.173.02/nvidia/nv-kernel.o_binary")


@dataclass(frozen=True)
class Function:
    name: str
    address: int
    size: int
    data: bytes


def functions(path: Path) -> tuple[list[Function], object]:
    stream = path.open("rb")
    elf = ELFFile(stream)
    text = elf.get_section_by_name(".text")
    symtab = elf.get_section_by_name(".symtab")
    text_index = elf.get_section_index(".text")
    found: list[Function] = []
    for symbol in symtab.iter_symbols():
        if (
            symbol["st_info"]["type"] == "STT_FUNC"
            and symbol["st_shndx"] == text_index
            and symbol["st_size"] > 0
        ):
            address = int(symbol["st_value"])
            size = int(symbol["st_size"])
            found.append(
                Function(symbol.name, address, size, text.data()[address : address + size])
            )
    found.sort(key=lambda item: (item.address, item.size, item.name))
    return found, stream


def disassembler() -> Cs:
    engine = Cs(CS_ARCH_X86, CS_MODE_64)
    engine.detail = True
    return engine


def dump(function: Function, engine: Cs) -> None:
    print(f"{function.name} address={function.address:#x} size={function.size:#x}")
    for insn in engine.disasm(function.data, function.address):
        print(f"  {insn.address:08x}: {insn.bytes.hex(' '):<30} {insn.mnemonic:<9} {insn.op_str}")


def memory_displacements(instructions: Iterable) -> list[tuple[int, int, int]]:
    output: list[tuple[int, int, int]] = []
    for insn in instructions:
        for operand in insn.operands:
            if operand.type == X86_OP_MEM:
                output.append((operand.mem.base, operand.mem.index, operand.mem.disp))
    return output


def immediate_values(instructions: Iterable) -> set[int]:
    return {
        operand.imm
        for insn in instructions
        for operand in insn.operands
        if operand.type == X86_OP_IMM
    }


def find_gmmu(candidates: list[Function], engine: Cs) -> list[Function]:
    matches: list[Function] = []
    for function in candidates:
        if not 20 <= function.size <= 160:
            continue
        instructions = list(engine.disasm(function.data, function.address))
        displacements = {
            displacement
            for base, _index, displacement in memory_displacements(instructions)
            if base == X86_REG_RDI
        }
        # GMMU_FMT_PTE has fldAddrPeer, fldAddrSysmem and fldAddrVidmem at
        # these three distinctive offsets in exact R580.173.02.
        if {0x30, 0x70, 0xB0}.issubset(displacements):
            matches.append(function)
    return matches


def find_gp100_dispatch(candidates: list[Function], engine: Cs) -> list[Function]:
    matches: list[Function] = []
    for function in candidates:
        if not 35 <= function.size <= 200:
            continue
        instructions = list(engine.disasm(function.data, function.address))
        immediates = immediate_values(instructions)
        memory = memory_displacements(instructions)
        has_bus_vtable = any(
            base == X86_REG_RSI and 0x80 <= displacement <= 0x1000
            for base, _index, displacement in memory
        )
        mnemonics = {insn.mnemonic for insn in instructions}
        has_connection_mask = any(
            insn.mnemonic == "and"
            and any(op.type == X86_OP_IMM and op.imm == 0xF for op in insn.operands)
            for insn in instructions
        )
        if (
            has_bus_vtable
            and has_connection_mask
            and 0x56 in immediates
            and 1 in immediates
            and "ret" in mnemonics
        ):
            matches.append(function)
    return matches


def find_phys_fb_addr(candidates: list[Function], engine: Cs) -> list[Function]:
    """Find gpumgrGetGpuPhysFbAddr-like leaf routines by control-flow shape."""
    matches: list[Function] = []
    for function in candidates:
        if not 25 <= function.size <= 100:
            continue
        instructions = list(engine.disasm(function.data, function.address))
        memory = memory_displacements(instructions)
        gpu_qword_loads = [
            insn
            for insn in instructions
            if (
            insn.mnemonic == "mov"
            and len(insn.operands) == 2
            and insn.operands[0].type != X86_OP_MEM
            and insn.reg_name(insn.operands[0].reg) == "rax"
            and insn.operands[1].type == X86_OP_MEM
            and insn.operands[1].mem.base == X86_REG_RDI
            )
        ]
        gpu_byte_tests = [
            insn
            for insn in instructions
            if (
            insn.mnemonic in {"cmp", "test"}
            and any(
                operand.type == X86_OP_MEM and operand.mem.base == X86_REG_RDI
                for operand in insn.operands
            )
            )
        ]
        mnemonics = [insn.mnemonic for insn in instructions]
        if (
            len(gpu_qword_loads) == 1
            and instructions.index(gpu_qword_loads[0]) <= 4
            and mnemonics.count("ret") >= 1
            and "call" in mnemonics
            and "test" in mnemonics
        ):
            matches.append(function)
    return matches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("gmmu", "gp100", "physfb"))
    args = parser.parse_args()

    candidates, stream = functions(CLOSED)
    try:
        engine = disassembler()
        if args.kind == "gmmu":
            matches = find_gmmu(candidates, engine)
        elif args.kind == "gp100":
            matches = find_gp100_dispatch(candidates, engine)
        else:
            matches = find_phys_fb_addr(candidates, engine)
        print(f"candidates={len(candidates)} matches={len(matches)}")
        for function in matches:
            dump(function, engine)
    finally:
        stream.close()


if __name__ == "__main__":
    main()
