"""Shared bits for the provisioning tests: a fast jig that needs no C compiler, a dev manufacturer, a provisioned device directory."""
from pathlib import Path

from dataopen.ctl.sim import dev_vendor
from dataopen.provisioning import station as ST


class FastJig(ST.Jig):
    """Reports every check as passed without running the bridge rig (the real jig is exercised in test_prov_provision.py)."""

    def run(self):
        return [ST.Check(n, True, "ok", lim) for n, (_, lim) in ST.CHECK_DOCS.items()]


def vendor():
    signer, pub = dev_vendor()
    return ST.VendorHsm(signer), pub


def provisioned(tmp_path: Path, name: str = "dev", sku: str = "DO-1", hsm=None, serial_db=None, **kw):
    hsm = hsm or vendor()[0]
    d = tmp_path / name
    rep = ST.provision(d, hsm, sku=sku, jig=FastJig(), serial_db=serial_db, **kw)
    assert not rep.quarantined, rep.steps
    return d, hsm, rep
