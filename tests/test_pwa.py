"""The browser client (pwa/): its generated files are current and its own tests pass, including the ones that run the client's code
against the real gateway and the real C bridge core through `dataopen ctl serve-sim`. Needs Node (skipped without it)."""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from dataopen.ctl import manifest as M
from dataopen.ctl import pwa as W

ROOT = Path(__file__).resolve().parents[1]
PWA = ROOT / "pwa"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="no node")
needs_cc = pytest.mark.skipif(shutil.which("gcc") is None and shutil.which("cc") is None, reason="no C compiler")


def test_the_service_worker_and_version_are_current():
    assert W.is_current(PWA), "regenerate: dataopen ctl pwa-build"


def test_every_cached_file_is_listed_and_tests_are_not():
    files = W.pwa_files(PWA)
    assert "index.html" in files and "js/build.js" in files and "js/app.js" in files
    assert not any(f.startswith("tests/") or f == "sw.js" for f in files)
    assert len(W.version_of(PWA, files)) == 12


def test_the_version_follows_the_content(tmp_path):
    shutil.copytree(PWA, tmp_path / "pwa", ignore=shutil.ignore_patterns("tests"))
    root = tmp_path / "pwa"
    v1 = W.build(root)
    assert W.is_current(root) and W.build(root) == v1
    (root / "app.css").write_text((root / "app.css").read_text() + "\n/* changed */\n")
    assert not W.is_current(root)
    assert W.build(root) != v1 and W.is_current(root)


def test_the_committed_manifest_fixture_is_the_devices_manifest():
    assert json.loads((PWA / "tests" / "manifest.json").read_text(encoding="utf-8")) == M.default_manifest()


def test_the_page_matches_the_chunk_size_the_device_allows():
    from dataopen.ctl import protocol as P
    js = (PWA / "js" / "session.js").read_text()
    assert re.search(r"chunk = 180", js) and 180 <= P.CHUNK_MAX      # the default proposal fits the protocol's limit


@needs_node
def test_every_script_parses():
    for f in sorted(PWA.rglob("*.js")) + sorted(PWA.rglob("*.mjs")):
        if "node_modules" in f.parts:
            continue
        r = subprocess.run([NODE, "--check", str(f)], capture_output=True, text=True)
        assert r.returncode == 0, f"{f}: {r.stderr}"


def _node(env_extra=None, files=None, timeout=400):
    env = {**os.environ, **(env_extra or {})}
    files = files or sorted(str(p) for p in (PWA / "tests").glob("*.test.mjs"))
    cmd = [NODE, "--test", "--test-force-exit", "--test-concurrency=1", "--test-timeout=60000", *files]
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout, cwd=ROOT)


@needs_node
def test_the_clients_own_tests():
    pure = [str(p) for p in sorted((PWA / "tests").glob("*.test.mjs")) if p.name != "e2e.test.mjs"]
    r = _node(files=pure)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    assert re.search(r"# pass (\d+)", r.stdout) and int(re.search(r"# pass (\d+)", r.stdout).group(1)) > 60


@needs_node
@needs_cc
def test_the_client_against_the_real_gateway_and_bridge_core():
    p = subprocess.Popen([sys.executable, "-m", "dataopen.cli", "ctl", "serve-sim", "--port", "0", "--trial-s", "3"], cwd=ROOT,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    try:
        url = None
        end = time.time() + 90
        while time.time() < end and url is None:
            line = p.stdout.readline()
            if not line and p.poll() is not None:
                break
            m = re.search(r"http://127\.0\.0\.1:\d+/", line)
            url = m.group(0) if m else None
        assert url, "the simulator server did not start"
        r = _node({"DATAOPEN_SIM_URL": url}, files=[str(PWA / "tests" / "e2e.test.mjs"), str(PWA / "tests" / "ble.test.mjs")])
        assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
        assert "# skipped 0" in r.stdout, "the end-to-end tests were skipped"
        assert int(re.search(r"# pass (\d+)", r.stdout).group(1)) >= 20
    finally:
        p.terminate()
        try:
            p.wait(10)
        except subprocess.TimeoutExpired:
            p.kill()
