"""
renloc_core/decompiler/disassembler.py

Multi-version Python bytecode disassembler using xdis.
Supports Python 2.7 through Python 3.12+ bytecode streams, converting raw
PyCode.bytecode or compiled code objects into clean, structured instruction streams.
"""

from __future__ import annotations

import io
import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import xdis
import xdis.magics
import xdis.marsh

# Opcode category helper sets
CONDITIONAL_JUMP_NAMES = {
    "POP_JUMP_IF_FALSE",
    "POP_JUMP_IF_TRUE",
    "POP_JUMP_FORWARD_IF_FALSE",
    "POP_JUMP_FORWARD_IF_TRUE",
    "POP_JUMP_BACKWARD_IF_FALSE",
    "POP_JUMP_BACKWARD_IF_TRUE",
    "POP_JUMP_IF_NOT_NONE",
    "POP_JUMP_IF_NONE",
    "JUMP_IF_FALSE_OR_POP",
    "JUMP_IF_TRUE_OR_POP",
    "JUMP_IF_NOT_DEBUG",
    "FOR_ITER",
    "SETUP_LOOP",
    "SETUP_EXCEPT",
    "SETUP_FINALLY",
    "SETUP_WITH",
}

UNCONDITIONAL_JUMP_NAMES = {
    "JUMP_FORWARD",
    "JUMP_BACKWARD",
    "JUMP_ABSOLUTE",
    "JUMP_BACKWARD_NO_INTERRUPT",
}

RETURN_NAMES = {
    "RETURN_VALUE",
    "RETURN_CONST",
}

RAISE_NAMES = {
    "RAISE_VARARGS",
    "RERAISE",
}


@dataclass
class DisassembledInstruction:
    """Represents a single disassembled bytecode instruction."""

    offset: int
    opcode: int
    opname: str
    arg: Optional[int] = None
    argval: Any = None
    argrepr: str = ""
    starts_line: Optional[int] = None
    is_jump_target: bool = False
    positions: Optional[Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]] = None

    @property
    def is_jump(self) -> bool:
        """True if this instruction can change control flow via a jump."""
        return (
            self.opname in CONDITIONAL_JUMP_NAMES
            or self.opname in UNCONDITIONAL_JUMP_NAMES
            or "JUMP" in self.opname
            or self.opname == "FOR_ITER"
        )

    @property
    def is_conditional_jump(self) -> bool:
        """True if this instruction conditionally jumps."""
        return (
            self.opname in CONDITIONAL_JUMP_NAMES
            or ("JUMP_IF" in self.opname)
            or ("POP_JUMP" in self.opname)
            or (self.opname == "FOR_ITER")
        )

    @property
    def is_unconditional_jump(self) -> bool:
        """True if this instruction always jumps unconditionally."""
        return self.opname in UNCONDITIONAL_JUMP_NAMES

    @property
    def is_return(self) -> bool:
        """True if this instruction returns from the frame."""
        return self.opname in RETURN_NAMES

    @property
    def is_terminal(self) -> bool:
        """True if this instruction terminates sequential execution of the basic block."""
        return self.is_return or self.is_unconditional_jump or (self.opname in RAISE_NAMES)

    @property
    def jump_target(self) -> Optional[int]:
        """Calculates or retrieves the target byte offset if this instruction is a jump."""
        if not self.is_jump:
            return None
        if isinstance(self.argval, int):
            return self.argval
        # Parse from argrepr (e.g. 'to 34')
        if self.argrepr and "to " in self.argrepr:
            try:
                parts = self.argrepr.split("to ")
                return int(parts[1].split()[0])
            except (ValueError, IndexError):
                pass
        return None

    def __repr__(self) -> str:
        line_str = f"L.{self.starts_line:<3d} " if self.starts_line else "       "
        target_marker = ">>" if self.is_jump_target else "  "
        arg_str = f"{self.arg:<5d}" if self.arg is not None else "     "
        val_str = f"({self.argval!r})" if self.argval is not None else ""
        return f"{line_str}{target_marker} {self.offset:>4d} {self.opname:<24s} {arg_str} {val_str}"


@dataclass
class DisassembledCode:
    """Represents a disassembled code object or script chunk."""

    version: Tuple[int, int]
    name: str
    filename: str
    firstlineno: int
    instructions: List[DisassembledInstruction] = field(default_factory=list)
    constants: Tuple[Any, ...] = ()
    names: Tuple[str, ...] = ()
    varnames: Tuple[str, ...] = ()
    freevars: Tuple[str, ...] = ()
    cellvars: Tuple[str, ...] = ()
    argcount: int = 0
    nested_codes: List[DisassembledCode] = field(default_factory=list)
    raw_code: Any = None

    def to_text(self) -> str:
        """Returns standard formatted disassembly text."""
        header = f"# Disassembly for {self.name} ({self.filename}:{self.firstlineno}) - Python {self.version[0]}.{self.version[1]}\n"
        body = "\n".join(repr(inst) for inst in self.instructions)
        nested = ""
        if self.nested_codes:
            nested = "\n\n" + "\n\n".join(nc.to_text() for nc in self.nested_codes)
        return header + body + nested

    def get_instruction_at(self, offset: int) -> Optional[DisassembledInstruction]:
        """Finds instruction at exact byte offset."""
        for inst in self.instructions:
            if inst.offset == offset:
                return inst
        return None

    def get_jump_targets(self) -> Set[int]:
        """Returns set of all target byte offsets in this code."""
        targets = set()
        for inst in self.instructions:
            t = inst.jump_target
            if t is not None:
                targets.add(t)
        return targets


class Disassembler:
    """Multi-version Python bytecode disassembler using xdis."""

    def __init__(self, default_version: Optional[Tuple[int, int]] = None):
        self.default_version = default_version or (sys.version_info[0], sys.version_info[1])

    @staticmethod
    def normalize_version(version: Any) -> Tuple[int, int]:
        """Normalizes various version formats into (major, minor)."""
        if isinstance(version, (tuple, list)):
            return int(version[0]), int(version[1])
        if isinstance(version, (float, int)):
            s = str(version)
            parts = s.split(".")
            if len(parts) == 1:
                return int(parts[0]), 0
            return int(parts[0]), int(parts[1])
        if isinstance(version, str):
            parts = version.strip().split(".")
            return int(parts[0]), int(parts[1])
        return (sys.version_info[0], sys.version_info[1])

    @classmethod
    def detect_version(cls, data: Any, default_version: Tuple[int, int] = (3, 12)) -> Tuple[int, int]:
        """Detects Python version from magic bytes or code object metadata."""
        if hasattr(data, "version_triple") and getattr(data, "version_triple"):
            vt = getattr(data, "version_triple")
            return int(vt[0]), int(vt[1])
        if hasattr(data, "python_version") and getattr(data, "python_version"):
            return cls.normalize_version(getattr(data, "python_version"))

        if isinstance(data, (bytes, bytearray)):
            if len(data) >= 4:
                magic_4 = bytes(data[:4])
                # Check known magic numbers in xdis
                for ver_str, m_bytes in xdis.magics.magics.items():
                    if magic_4 == m_bytes:
                        try:
                            return cls.normalize_version(ver_str)
                        except Exception:
                            continue

        if isinstance(data, types.CodeType):
            return sys.version_info[0], sys.version_info[1]

        return default_version

    def _get_opc_module(self, version: Tuple[int, int]):
        """Retrieves opcode module from xdis for target version."""
        try:
            return xdis.get_opcode(version, "CPython")
        except Exception:
            # Fallback to closest major version
            if version[0] == 2:
                return xdis.get_opcode((2, 7), "CPython")
            else:
                return xdis.get_opcode((3, 12), "CPython")

    def disassemble(
        self,
        target: Any,
        python_version: Optional[Any] = None,
    ) -> DisassembledCode:
        """
        Disassembles target into a DisassembledCode structure.

        Args:
            target: PyCode, types.CodeType, xdis Code object, bytes, or file path.
            python_version: Optional version override (e.g. (2, 7), (3, 12), '2.7', 3.12).

        Returns:
            DisassembledCode containing instruction stream and metadata.
        """
        # Resolve PyCode objects (from renloc_core.ast_reader.ast_nodes)
        if hasattr(target, "bytecode") or hasattr(target, "source"):
            if getattr(target, "bytecode", None) is not None:
                target = getattr(target, "bytecode")
            elif getattr(target, "source", None):
                src = getattr(target, "source")
                loc = getattr(target, "location", ("<renpy>", 1))
                mode = getattr(target, "mode", "exec")
                target = compile(src, loc[0] if loc else "<renpy>", mode)

        # Resolve Path / string file path
        if isinstance(target, (str, Path)) and os.path.exists(str(target)):
            with open(str(target), "rb") as f:
                target = f.read()

        # Determine version
        ver: Tuple[int, int]
        if python_version is not None:
            ver = self.normalize_version(python_version)
        else:
            ver = self.detect_version(target, self.default_version)

        code_obj = None

        if isinstance(target, (bytes, bytearray)):
            target_bytes = bytes(target)
            # Try 1: check if it's a full .pyc file with magic header
            if len(target_bytes) >= 4 and any(target_bytes.startswith(m) for m in xdis.magics.magics.values()):
                try:
                    res = xdis.load_module_from_file_object(io.BytesIO(target_bytes))
                    detected_ver = (res[0][0], res[0][1])
                    if python_version is None:
                        ver = detected_ver
                    code_obj = res[3]
                except Exception:
                    code_obj = None

            # Try 2: unmarshal code object using xdis.marsh
            if code_obj is None:
                try:
                    code_obj = xdis.marsh.loads(target_bytes, python_version=ver)
                except Exception:
                    code_obj = None

            # Try 3: native marshal loads
            if code_obj is None:
                import marshal
                try:
                    code_obj = marshal.loads(target_bytes)
                except Exception:
                    code_obj = None

            # Try 4: raw co_code bytes wrapper
            if code_obj is None:
                code_obj = self._create_dummy_code(target_bytes, ver)

        elif isinstance(target, types.CodeType) or hasattr(target, "co_code"):
            code_obj = target
        else:
            raise TypeError(f"Unsupported disassembly target type: {type(target)}")

        return self._disassemble_code_object(code_obj, ver)

    def _create_dummy_code(self, co_code: bytes, version: Tuple[int, int]) -> Any:
        """Creates a portable code representation when only raw bytecodes are given."""
        if version[0] == 2:
            return xdis.Code2(
                co_argcount=0,
                co_nlocals=0,
                co_stacksize=4,
                co_flags=64,
                co_code=co_code,
                co_consts=(),
                co_names=(),
                co_varnames=(),
                co_freevars=(),
                co_cellvars=(),
                co_filename="<raw_bytecode>",
                co_name="<module>",
                co_firstlineno=1,
                co_lnotab=b"",
            )
        else:
            class RawCodeHolder:
                pass

            holder = RawCodeHolder()
            holder.co_code = co_code
            holder.co_consts = ()
            holder.co_names = ()
            holder.co_varnames = ()
            holder.co_freevars = ()
            holder.co_cellvars = ()
            holder.co_firstlineno = 1
            holder.co_lnotab = b""
            holder.co_argcount = 0
            holder.co_posonlyargcount = 0
            holder.co_kwonlyargcount = 0
            holder.co_nlocals = 0
            holder.co_stacksize = 4
            holder.co_flags = 64
            holder.co_filename = "<raw_bytecode>"
            holder.co_name = "<module>"
            return holder

    def _disassemble_code_object(self, code_obj: Any, version: Tuple[int, int]) -> DisassembledCode:
        """Disassembles an unmarshalled code object or portable code into DisassembledCode."""
        opc = self._get_opc_module(version)
        name = getattr(code_obj, "co_name", "<unknown>")
        filename = getattr(code_obj, "co_filename", "<unknown>")
        firstlineno = getattr(code_obj, "co_firstlineno", 1)
        constants = tuple(getattr(code_obj, "co_consts", ()))
        names = tuple(getattr(code_obj, "co_names", ()))
        varnames = tuple(getattr(code_obj, "co_varnames", ()))
        freevars = tuple(getattr(code_obj, "co_freevars", ()))
        cellvars = tuple(getattr(code_obj, "co_cellvars", ()))
        argcount = getattr(code_obj, "co_argcount", 0)

        instructions: List[DisassembledInstruction] = []
        try:
            bc = xdis.Bytecode(code_obj, opc)
            for inst in bc:
                dis_inst = DisassembledInstruction(
                    offset=inst.offset,
                    opcode=inst.opcode,
                    opname=inst.opname,
                    arg=inst.arg,
                    argval=inst.argval,
                    argrepr=inst.argrepr,
                    starts_line=inst.starts_line,
                    is_jump_target=inst.is_jump_target,
                    positions=getattr(inst, "positions", None),
                )
                instructions.append(dis_inst)
        except Exception:
            # Fallback to get_instructions_bytes if Bytecode() encountered version divergence
            try:
                for inst in xdis.get_instructions_bytes(code_obj, opc):
                    dis_inst = DisassembledInstruction(
                        offset=inst.offset,
                        opcode=inst.opcode,
                        opname=inst.opname,
                        arg=inst.arg,
                        argval=inst.argval,
                        argrepr=inst.argrepr,
                        starts_line=inst.starts_line,
                        is_jump_target=inst.is_jump_target,
                        positions=getattr(inst, "positions", None),
                    )
                    instructions.append(dis_inst)
            except Exception as e:
                # Minimal fallback: emit error instruction
                instructions.append(
                    DisassembledInstruction(
                        offset=0,
                        opcode=0,
                        opname=f"DISASSEMBLY_ERROR_{e.__class__.__name__}",
                        argrepr=str(e),
                    )
                )

        # Normalize jump targets across all instructions
        target_offsets = set()
        for inst in instructions:
            t = inst.jump_target
            if t is not None:
                target_offsets.add(t)

        for inst in instructions:
            if inst.offset in target_offsets:
                inst.is_jump_target = True

        # Process nested code objects recursively
        nested_codes: List[DisassembledCode] = []
        for const in constants:
            if isinstance(const, types.CodeType) or hasattr(const, "co_code"):
                try:
                    nested = self._disassemble_code_object(const, version)
                    nested_codes.append(nested)
                except Exception:
                    continue

        return DisassembledCode(
            version=version,
            name=name,
            filename=filename,
            firstlineno=firstlineno,
            instructions=instructions,
            constants=constants,
            names=names,
            varnames=varnames,
            freevars=freevars,
            cellvars=cellvars,
            argcount=argcount,
            nested_codes=nested_codes,
            raw_code=code_obj,
        )
