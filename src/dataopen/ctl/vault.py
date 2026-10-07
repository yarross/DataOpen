"""The device's own files on disk, encrypted under the device's storage key (docs/SECURITY.md section 5).

    'DOVT' | ver=1 | nonce[12] | ChaCha20-Poly1305(data, associated data = 'DOVT1' || file name)

The associated data pins a blob to its file name, so a sealed settings file can not be dropped in where the profile belongs. Losing the
storage key (`Identity.rotate_storage_key`) makes every older blob unreadable at once: that is the 'crypto-erase' of the personal profile,
and it works even where the flash keeps old blocks that an overwrite would not reach.
"""
from __future__ import annotations

import os
from pathlib import Path

MAGIC, VERSION = b"DOVT", 1


class VaultError(ValueError):
    pass


class Vault:
    def __init__(self, key: bytes) -> None:
        if len(key) != 32:
            raise ValueError("the storage key is 32 bytes")
        self.key = key

    @staticmethod
    def is_sealed(blob: bytes) -> bool:
        return blob[:4] == MAGIC

    def seal(self, name: str, data: bytes) -> bytes:
        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
        nonce = os.urandom(12)
        return MAGIC + bytes([VERSION]) + nonce + ChaCha20Poly1305(self.key).encrypt(nonce, data, b"DOVT1" + name.encode())

    def open(self, name: str, blob: bytes) -> bytes:
        from cryptography.exceptions import InvalidTag
        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
        if len(blob) < 4 + 1 + 12 + 16 or blob[:4] != MAGIC or blob[4] != VERSION:
            raise VaultError("not a sealed file")
        try:
            return ChaCha20Poly1305(self.key).decrypt(blob[5:17], blob[17:], b"DOVT1" + name.encode())
        except InvalidTag:
            raise VaultError("wrong key, wrong file or damaged") from None

    def migrate(self, path: Path, name: str) -> bool:
        """Seal a plain file in place (temp file + atomic rename). True if it was plain."""
        try:
            raw = path.read_bytes()
        except OSError:
            return False
        if self.is_sealed(raw):
            return False
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(self.seal(name, raw))
        os.replace(tmp, path)
        return True

    def read(self, path: Path, name: str, allow_plain: bool = True) -> bytes:
        """The file's content. A plain (older) file is accepted once, so a device that predates the vault migrates on its next write."""
        raw = path.read_bytes()
        if self.is_sealed(raw):
            return self.open(name, raw)
        if not allow_plain:
            raise VaultError("not sealed")
        return raw
