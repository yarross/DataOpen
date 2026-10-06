"""Known-good inputs (real descriptors, a composite configuration, parameter blobs) written as a C header for the standalone C programs
that exercise the core: the sanitizer fuzz driver (tests/helpers/bridge_fuzz.c) and the timing benchmark (csrc/bench_main.c)."""

from __future__ import annotations

from pathlib import Path

from ..assist.fixed import FixedParams
from ..assist.tremor_fixed import FixedTremorParams
from . import protocol as P
from . import sim_usb as U


def _arr(name: str, data: bytes) -> str:
    return f"static const uint8_t {name}[] = {{" + ",".join(str(b) for b in data) + "};\n"


def write_seeds_header(path: Path, asc: FixedParams, tremor: FixedTremorParams) -> None:
    m = U.SimMouse("logi", hidpp=True, kbd=True)
    asc_blob = P.make_blob(1, 1, 65536, P.asc_blob_bytes(asc))
    trm_blob = P.make_blob(1, 1, 0, P.tremor_blob_bytes(tremor))
    text = (
        "#include <stdint.h>\n"
        + _arr("SEED_RD_BOOT", U.boot_mouse_rd())
        + _arr("SEED_RD_M16", U.mouse16_rd())
        + _arr("SEED_RD_LOGI", U.logi_mouse_rd())
        + _arr("SEED_RD_HIDPP", U.hidpp_rd())
        + _arr("SEED_RD_KBD", U.kbd_rd())
        + _arr("SEED_CFG", m.cfg)
        + _arr("SEED_DEV", m.dev)
        + _arr("SEED_ASC_BLOB", asc_blob)
        + _arr("SEED_TRM_BLOB", trm_blob)
        + f"#define ASC_BLOB_LEN {len(asc_blob)}\n#define TRM_BLOB_LEN {len(trm_blob)}\n"
    )
    Path(path).write_text(text)
