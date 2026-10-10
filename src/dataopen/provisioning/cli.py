"""`dataopen prov ...` (the factory) and `dataopen recover ...` (support): provisioning, the chain of trust, recovery scenarios and the full return."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from . import recovery as RC
from . import records as R
from . import report as RP
from . import service as SV
from . import station as ST
from .device import DeviceAgent, DeviceError

EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 4
DOC = Path(__file__).resolve().parents[3] / "docs" / "PROVISIONING.md"
FAULTS = [f for f in ST.BoardFaults.__dataclass_fields__ if f != "relay_slow_ms"]


def _dev_hsm() -> ST.VendorHsm:
    from ..ctl.sim import dev_vendor
    return ST.VendorHsm(dev_vendor()[0])


def _provision(a) -> int:
    d = Path(a.dir) if a.dir else Path(tempfile.mkdtemp(prefix="dataopen-prov-"))
    faults = ST.BoardFaults(**{f: True for f in (a.fault or [])})
    try:
        rep = ST.provision(d, _dev_hsm(), sku=a.sku, jig=ST.Jig(faults))
    except KeyError:
        print(f"error: no such SKU: {a.sku}")
        return EXIT_USAGE
    if a.json:
        print(json.dumps({**rep.to_json(), "label": rep.label, "digest": rep.digest, "dir": str(d)}, ensure_ascii=False, indent=2))
    else:
        print(f"device directory: {d}")
        for s in rep.steps:
            print(f"  {'ok  ' if s.ok else 'FAIL'} {s.name:9} {s.detail}")
        for c in rep.checks:
            print(f"       {'ok  ' if c.ok else 'FAIL'} {c.name:22} {c.measured}  (limit: {c.limit})")
        print("QUARANTINED at " + rep.failed_step if rep.quarantined else f"label: {rep.label['serial']}  (owner ID at birth {rep.label['birth_id']})")
    return EXIT_FAILED if rep.quarantined else EXIT_OK


def _agent(a) -> DeviceAgent:
    from ..ctl.sim import dev_vendor
    return DeviceAgent(a.dir, hw_id=ST.DEFAULT_HW, vendor_pub=dev_vendor()[1])


def _verify(a) -> int:
    from ..ctl.identity import Card
    ag = _agent(a)
    rec = ag.record
    if rec is None:
        print("not provisioned")
        return EXIT_FAILED
    owner = ag.owner()
    cj = owner.card().to_json()
    cj["device"] = ag.cert().to_json()
    try:
        serial = R.verify_chain(Card.from_json(cj), ag.vendor_pub, ag.hw_id)
    except R.RecordError as e:
        print(f"chain broken: {e.key}")
        return EXIT_FAILED
    print(f"serial {serial}, lifecycle {ag.lifecycle}, owner ID {owner.id}: manufacturer -> attestation -> DAK -> owner card verified")
    return EXIT_OK


def _label(a) -> int:
    ag = _agent(a)
    rec = ag.record
    if rec is None:
        print("not provisioned")
        return EXIT_FAILED
    cj = ag.owner().card().to_json()
    cj["device"] = ag.cert().to_json()
    print(json.dumps({"serial": rec.serial, "sku": rec.sku, "qr": ST.qr_payload(cj)}, ensure_ascii=False, indent=2))
    return EXIT_OK


def _table(fn, data=None):
    def run(a) -> int:
        print(json.dumps(data(), ensure_ascii=False, indent=2) if (a.json and data) else fn())
        return EXIT_OK
    return run


def _docs(a) -> int:
    path = Path(a.path) if a.path else DOC
    if a.check:
        if not RP.doc_is_current(path):
            print(f"{path} is stale: run `dataopen prov docs --write`")
            return EXIT_FAILED
        print("current")
        return EXIT_OK
    path.write_text(RP.render_doc(path.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"written: {path}")
    return EXIT_OK


def _simulate(a) -> int:
    chosen = [s for s in RC.SCENARIOS if a.scenario in (None, "all", s.key)]
    if not chosen:
        print(f"error: no such scenario; one of: {', '.join(s.key for s in RC.SCENARIOS)}")
        return EXIT_USAGE
    for s in chosen:
        o = RC.run(s)
        if a.json:
            print(json.dumps({"scenario": s.key, "first": vars(o.first), "final": vars(o.final), "computer": o.computer, "profiles_kept": o.profiles_kept,
                              "rma": o.rma, "steps": o.steps}, ensure_ascii=False))
            continue
        print(f"{s.key}: {s.event}")
        print(f"  right away: {o.first.stage}, mouse {o.first.mouse}, video {o.first.video}, LED {o.first.led}")
        for n in o.first.notes:
            print(f"    - {n}")
        print("  way back: " + "; ".join(t for _, t in s.procedure))
        print(f"  computer: {'needed' if o.computer else 'not needed'}; profiles: "
              f"{'kept' if o.profiles_kept else 'lost, return to the manufacturer' if o.rma else 'lost: the person calibrates again (there is no copy)'}")
    return EXIT_OK


def _full_return(a) -> int:
    ag = _agent(a)
    rec = ag.record
    if rec is None:
        print("not provisioned")
        return EXIT_FAILED
    hsm = _dev_hsm()
    hsm.shipped.add(rec.serial)                      # the dev desk trusts the directory; a real desk looks the serial up in its database
    try:
        tok = hsm.issue_service_token(rec.serial, "factory_return", ag.service_challenge())
        out = SV.full_return(a.dir, ag, tok.to_json(), presence=a.hand_on_device)
    except (ST.HsmError, SV.ServiceError, DeviceError) as e:
        print(f"refused: {e.key}: {e}")
        return EXIT_FAILED
    print(json.dumps(out, indent=2))
    return EXIT_OK


def register(sub) -> None:
    p = sub.add_parser("prov", help="factory provisioning: serial, device keys, attestation, checks, label (docs/PROVISIONING.md)")
    ps = p.add_subparsers(dest="prov_cmd", required=True)
    v = ps.add_parser("provision", help="provision one simulated board end to end")
    v.add_argument("--dir")
    v.add_argument("--sku", default="DO-1")
    v.add_argument("--fault", action="append", choices=FAULTS, help="inject a board fault to see the jig catch it")
    v.add_argument("--json", action="store_true")
    v.set_defaults(fn=_provision)
    for name, fn, h in (("verify", _verify, "check the chain manufacturer -> DAK -> owner card of a provisioned directory"),
                        ("label", _label, "the label data (serial, QR payload)")):
        x = ps.add_parser(name, help=h)
        x.add_argument("--dir", required=True)
        x.set_defaults(fn=fn)
    for name, fn, data, h in (("steps", RP.steps_table, None, "the provisioning steps"), ("checks", RP.checks_table, None, "the jig's checks"),
                              ("identity", RP.identity_table, None, "what is immutable and what can be reset"),
                              ("storage", RP.storage_table, lambda: [vars(i) for i in R.ITEMS], "where each piece of state lives"),
                              ("levels", RP.levels_table, lambda: [dict(zip(("level", "name", "how", "what"), x)) for x in R.LEVELS], "the four levels of reset"),
                              ("lifecycle", RP.lifecycle_table, None, "the lifecycle")):
        x = ps.add_parser(name, help=h)
        x.add_argument("--json", action="store_true")
        x.set_defaults(fn=_table(fn, data))
    g = ps.add_parser("docs", help="refresh (or check) the generated tables in docs/PROVISIONING.md")
    g.add_argument("--path")
    g.add_argument("--check", action="store_true")
    g.add_argument("--write", action="store_true")
    g.set_defaults(fn=_docs)
    r = sub.add_parser("recover", help="recovery and the full return to factory state (support; docs/PROVISIONING.md section 4)")
    rs = r.add_subparsers(dest="recover_cmd", required=True)
    s = rs.add_parser("simulate", help="walk a recovery scenario: what the person sees, the way back, the computer, the profiles")
    s.add_argument("scenario", nargs="?", help="a scenario key or 'all'")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=_simulate)
    t = rs.add_parser("scenarios", help="the table of scenarios")
    t.set_defaults(fn=_table(RP.scenarios_table), json=False)
    f = rs.add_parser("full-return", help="L4: the full return to factory state of a simulated device directory (dev manufacturer key)")
    f.add_argument("--dir", required=True)
    f.add_argument("--hand-on-device", action="store_true", help="assert the physical presence (the recovery chord or the RECOVERY button)")
    f.set_defaults(fn=_full_return)
