import json
import threading
import time
from pathlib import Path

import pytest

from dataopen.adapters.mock import MockGameAdapter
from dataopen.adapters.mock.server import MockServerOptions, serve_mock
from dataopen.adapters.remote import RemoteGameAdapter, RemoteOptions
from dataopen.cli import EXIT_OK, EXIT_USAGE, main
from dataopen.core.orchestrator import DatasetOrchestrator, SessionAborted, SessionConfig
from dataopen.core.qa import verify_dataset
from dataopen.core.randomization import DomainRandomizationController
from dataopen.core.transport import FileMailboxTransport
from dataopen.quality.evaluators.simulated import CallableEvaluator, SimulatedEvaluator
from dataopen.quality.feedback import AdaptiveRandomizer, FeedbackConfig
from dataopen.quality.pipeline import QualityConfig, QualityPipeline
from dataopen.core.schema import HUMAN_13


def records(root: Path):
    out = []
    for p in sorted((root / "annotations").glob("*.jsonl")):
        out += [json.loads(line) for line in p.read_text().splitlines()]
    return out


def pipeline(evaluator=None, **kw):
    return QualityPipeline(HUMAN_13, evaluator if evaluator is not None else SimulatedEvaluator(HUMAN_13), QualityConfig(**kw))


def run(out, frames=60, adapter=None, quality=None, randomizer=None, **cfg):
    adapter = adapter or MockGameAdapter()
    c = SessionConfig(out, seed=cfg.pop("seed", 3), target_frames=frames, frames_per_scene=cfg.pop("frames_per_scene", 6),
                      negative_ratio=cfg.pop("negative_ratio", 0.15), **cfg)
    return DatasetOrchestrator(adapter, c, randomizer=randomizer, quality=quality).run()


def test_closed_loop_end_to_end_artifacts_and_semantics(tmp_path):
    out = tmp_path / "ds"
    adapter = MockGameAdapter()
    rz = AdaptiveRandomizer(3, adapter.parameter_space(), feedback=FeedbackConfig(warmup_frames=40, update_every=10))
    rep = run(out, 90, adapter, pipeline(), rz)
    assert rep.accepted == 90 and rep.stop_reason == "target_reached"
    recs = records(out)
    assert len(recs) == 90 and all("quality" in r["meta"] for r in recs)
    tiers = {r["meta"]["quality"]["tier"] for r in recs}
    assert tiers <= {"keep", "keep_hard", "hard_negative"} and "keep" in tiers
    # every record carries the difficulty metadata and the unit draws that produced it (replay)
    q = recs[0]["meta"]["quality"]
    assert set(q) >= {"difficulty", "occlusion_index", "contrast_rate", "weight", "metrics", "features", "tier"}
    assert recs[0]["meta"]["units"]["frame"]["cam"] and "actors" in recs[0]["meta"]
    pos = next(r for r in recs if r["annotations"])
    assert {"occlusion_index", "contrast_rate", "perceptibility", "oks"} <= set(pos["annotations"][0]["meta"])
    # dropped frames leave evidence but no images; accepted frames have exactly one image
    dropped = {k: v for k, v in rep.rejects.items() if k.startswith("quality_")}
    assert dropped and sum(dropped.values()) > 0
    assert len(list(out.glob("images/*/*.png"))) == 90
    lines = (out / "quality" / "rejects.jsonl").read_text().splitlines()
    assert len(lines) == sum(dropped.values()) and json.loads(lines[0])["tier"].startswith(("drop_", "suspect"))
    # reports and exports
    assert (out / "closed_loop_report.md").exists() and (out / "feedback_state.json").exists()
    idx = json.loads((out / "quality_index.json").read_text())
    assert len(idx) == 90 and all(v["weight"] >= 1.0 for v in idx.values())
    coco = json.loads(next(out.glob("annotations/coco_train.json")).read_text())
    assert "difficulty" in coco["images"][0] and "tier" in coco["images"][0]
    assert any("oks" in a and "occlusion_index" in a for a in coco["annotations"])
    v = verify_dataset(out)
    assert v.ok and v.stats["quality"]["tiers"] and v.stats["quality"]["evaluated_share"] > 0.3
    md = (out / "closed_loop_report.md").read_text()
    assert "Closed-loop validation report" in md and "bias" in md.lower() and "Adaptive randomization" in md
    assert rep.quality["pipeline"]["evaluated_by_model"] > 0


def test_cheap_gates_drop_unusable_frames_without_running_the_model(tmp_path):
    class Night(DomainRandomizationController):
        def sample_scene(self, i):
            s = super().sample_scene(i)
            s.environment.update(time_of_day=1.0, weather="clear", fog_density=0.0004, cloud_cover=0.1)
            return s

    calls = []
    ev = CallableEvaluator(lambda imgs: calls.append(1) or [[] for _ in imgs])
    adapter = MockGameAdapter()
    q = pipeline(ev)
    rep = run(tmp_path / "ds", 12, adapter, q, Night(3, adapter.parameter_space()), max_attempt_factor=6)
    st = rep.quality["pipeline"]
    assert st["dropped_by_cheap_gates"] > 0 and st["evaluated_by_model"] < st["frames"]
    assert len(calls) == st["evaluated_by_model"]                          # inference saved on every gated frame
    assert any(k in st["tiers"] for k in ("drop_invisible", "drop_render"))


def test_inference_overlaps_with_capture(tmp_path):
    state = {"running": 0, "max_running_seen_by_capture": 0, "capture_while_eval": 0}
    lock = threading.Lock()

    def slow(imgs):
        with lock:
            state["running"] += 1
        time.sleep(0.04)
        with lock:
            state["running"] -= 1
        return [[] for _ in imgs]

    class Watching(MockGameAdapter):
        def __init__(self):
            super().__init__()
            real = self.capture.capture

            def capture(req):
                with lock:
                    if state["running"] > 0:
                        state["capture_while_eval"] += 1
                return real(req)
            self.capture.capture = capture

    run(tmp_path / "ds", 30, Watching(), pipeline(CallableEvaluator(slow), max_inflight=3))
    assert state["capture_while_eval"] > 3                                 # the engine rendered while the model ran


def test_one_off_evaluator_failure_is_survived_but_repeated_failure_stops_the_session(tmp_path):
    n = {"i": 0}

    def flaky(imgs):
        n["i"] += 1
        if n["i"] == 4:
            raise RuntimeError("CUDA out of memory")
        return [[] for _ in imgs]

    rep = run(tmp_path / "a", 20, quality=pipeline(CallableEvaluator(flaky)))
    assert rep.accepted == 20 and rep.quality["pipeline"]["evaluator_errors"] == 1

    def broken(imgs):
        raise RuntimeError("model file corrupted")

    with pytest.raises(SessionAborted, match="evaluator keeps failing.*model file corrupted"):
        run(tmp_path / "b", 20, quality=pipeline(CallableEvaluator(broken)))
    assert json.loads((tmp_path / "b" / "report.json").read_text())["stop_reason"] == "aborted"


class Game:
    def __init__(self, directory, **opts):
        self.stop = threading.Event()
        self.t = threading.Thread(target=serve_mock, args=(directory, MockServerOptions(**opts), self.stop), daemon=True)
        self.t.start()

    def close(self):
        self.stop.set()
        self.t.join(3)


@pytest.mark.parametrize("shm", [True, False])
def test_closed_loop_over_the_wire_uses_shared_memory_or_a_staged_file(tmp_path, shm):
    mb = tmp_path / "mb"
    g = Game(mb, image_shm=shm)
    try:
        a = RemoteGameAdapter(FileMailboxTransport(mb, default_timeout_s=10), RemoteOptions())
        a.connect()
        free_before = a.ring.free_slots if shm else None
        rep = run(tmp_path / "ds", 40, a, pipeline(max_inflight=2))
        assert rep.accepted == 40 and rep.quality["pipeline"]["frames_without_pixels"] == 0
        assert verify_dataset(tmp_path / "ds").ok
        if shm:
            assert free_before == a.options.shm_slots
        assert not list((mb / "staging").glob("peek_*"))
    finally:
        g.close()


def test_resume_restores_the_feedback_state(tmp_path):
    out = tmp_path / "ds"
    adapter = MockGameAdapter()
    mk = lambda: AdaptiveRandomizer(3, adapter.parameter_space(), feedback=FeedbackConfig(warmup_frames=30, update_every=10))  # noqa: E731
    r1 = mk()
    run(out, 60, adapter, pipeline(), r1)
    seen = r1.n_observed
    assert seen >= 60 and r1.adapting
    r2 = mk()
    run(out, 100, MockGameAdapter(), pipeline(), r2, resume=True)
    assert r2.n_observed > seen                                           # continued from the checkpoint, not from zero
    assert verify_dataset(out).ok


def test_adaptive_run_wastes_fewer_attempts_than_blind_sampling(tmp_path):
    """Measured on the mock over 3 seeds: ~25% fewer attempts for the same number of accepted frames (the share of hard
    frames also rises on average, but it is noisy per run, so only the robust effect is asserted)."""
    def attempts(adaptive: bool) -> int:
        adapter = MockGameAdapter()
        rz = (AdaptiveRandomizer(22, adapter.parameter_space(), feedback=FeedbackConfig(warmup_frames=60, update_every=10))
              if adaptive else None)
        return run(tmp_path / ("a" if adaptive else "b"), 300, adapter, pipeline(), rz, seed=22, frames_per_scene=8).attempts

    adaptive, blind = attempts(True), attempts(False)
    assert adaptive < 0.95 * blind


def test_validator_rejections_are_fed_back_as_wasted_attempts():
    from dataopen.quality.types import Tier, rejected_verdict
    v = rejected_verdict("unlabeled_person_present")
    assert v.tier is Tier.REJECTED and v.tier.is_drop and v.utility == 0.0 and v.weight == 0.0


def test_cli_collect_with_simulated_quality_and_adaptive_and_static(tmp_path, capsys):
    out = tmp_path / "ds"
    with pytest.raises(SystemExit) as e:
        main(["collect", "--game", "mock", "--out", str(out), "--frames", "40", "--quality-sim", "--adaptive"])
    assert e.value.code == EXIT_OK and (out / "closed_loop_report.md").exists()
    with pytest.raises(SystemExit) as e:
        main(["collect", "--game", "mock", "--out", str(tmp_path / "s"), "--frames", "20", "--quality-static"])
    assert e.value.code == EXIT_OK
    st = json.loads((tmp_path / "s" / "closed_loop_report.json").read_text())["pipeline"]
    assert st["backend"] is None and st["evaluated_by_model"] == 0
    with pytest.raises(SystemExit) as e:
        main(["collect", "--game", "mock", "--out", str(tmp_path / "m"), "--frames", "5", "--quality-model",
              str(tmp_path / "missing.onnx")])
    assert e.value.code == EXIT_USAGE and "quality subsystem" in capsys.readouterr().out
