#!/usr/bin/env python3
"""
tests/test_actor_transport.py — unit tests for the Python ActorTransport Port + SubprocessActorTransport,
driven against a FAKE runner that speaks the control_spec protocol in pure Python (no C++ build, no
redis). This is the test-seam analog of test_zmq_inference's fake sockets: it pins the transport's
encode/decode, the two gates (config_epoch / version), the loud error-tag translation, the BOUNDED-RECV
non-hang discipline, and the graceful/forceful reap — independent of the C++ runner (whose own behavior
is covered by the C++-build-gated integration parity in test_cpp_runner.py).

The fake runner DERIVES the protocol vocabulary from chocofarm/az/control_spec.py (loaded by file path,
no package __init__ chain), so it stays in lock-step with the SSOT the transport uses — a transport that
mis-spelled a key/tag would diverge from the fake and red here.

Public Domain (The Unlicense).
"""
import os
import stat
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chocofarm.az import control_spec as C
from chocofarm.az.actor_config import ActorConfig
from chocofarm.az.actor_transport import (
    ControlError, ERR_RECV_TIMEOUT, ERR_RUNNER_DIED, GenerateRequest, GenerateResult, PingResult,
    SubprocessActorTransport,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The fake runner: a stdlib-only script that speaks the control_spec protocol on stdin/stdout. It derives
# the keys/tags from control_spec.py (loaded by path), increments an epoch per configure, gates generate
# on (configured? epoch match?), rejects an instance_path change, and honors test hooks (--hang-generate
# never replies to generate; --die-on-generate exits without replying) to exercise the bounded recv.
_FAKE_BODY = '''
import sys, os, json, importlib.util
_spec = importlib.util.spec_from_file_location(
    "control_spec", os.path.join("__REPO__", "chocofarm", "az", "control_spec.py"))
C = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(C)

def reply(obj):
    sys.stdout.write(json.dumps(obj) + "\\n"); sys.stdout.flush()

args = sys.argv[1:]
hang_generate = "--hang-generate" in args
die_on_generate = "--die-on-generate" in args
epoch = 0
instance = None
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        reply({C.KEY_OK: False, C.KEY_ERROR: C.ERR_BAD_JSON, C.KEY_DETAIL: "bad json"}); continue
    t = msg.get(C.KEY_TYPE)
    if t == C.MSG_CONFIGURE:
        cfg = msg.get(C.KEY_CONFIG, {})
        inst = cfg.get("instance_path")
        if instance is not None and inst != instance:
            reply({C.KEY_OK: False, C.KEY_ERROR: C.ERR_INSTANCE_KNOB,
                   C.KEY_DETAIL: str(instance) + "->" + str(inst)}); continue
        instance = inst
        epoch += 1
        reply({C.KEY_OK: True, C.KEY_CONFIG_EPOCH: epoch})
    elif t == C.MSG_GENERATE:
        if hang_generate:
            continue
        if die_on_generate:
            sys.exit(0)
        if epoch == 0:
            reply({C.KEY_OK: False, C.KEY_ERROR: C.ERR_NOT_CONFIGURED,
                   C.KEY_DETAIL: "configure first"}); continue
        if msg.get(C.KEY_CONFIG_EPOCH) != epoch:
            reply({C.KEY_OK: False, C.KEY_ERROR: C.ERR_EPOCH_MISMATCH,
                   C.KEY_DETAIL: "have " + str(epoch)}); continue
        reply({C.KEY_OK: True, C.KEY_WRITTEN: msg.get(C.KEY_EPISODES, 0),
               C.KEY_CONFIG_EPOCH: epoch, C.KEY_VERSION: msg.get(C.KEY_VERSION)})
    elif t == C.MSG_PING:
        reply({C.KEY_OK: True, C.KEY_SERVING: epoch > 0, C.KEY_CONFIG_EPOCH: epoch})
    elif t == C.MSG_SHUTDOWN:
        reply({C.KEY_OK: True}); sys.exit(0)
    else:
        reply({C.KEY_OK: False, C.KEY_ERROR: C.ERR_UNKNOWN_TYPE, C.KEY_DETAIL: str(t)})
'''


def _write_fake_runner(tmp_path) -> str:
    """Write the fake runner to an executable script (shebang = this venv's python, so it can importlib-
    load control_spec) and return its path. The transport spawns it as `[path, "--serve", *extra_args]`."""
    script = "#!" + sys.executable + "\n" + _FAKE_BODY.replace("__REPO__", REPO)
    path = os.path.join(str(tmp_path), "fake_actor_runner.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(script)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IRUSR)
    return path


def _cfg(instance_path: str = "chocofarm/data/instance.json") -> ActorConfig:
    return ActorConfig(instance_path=instance_path, faces_path="chocofarm/data/faces.json",
                       m=12, n_sims=48, c_puct=1.25, c_visit=50.0, c_scale=1.0, c_outcome=2, max_depth=24)


def _req(config_epoch: int, version: int = 0, episodes: int = 7) -> GenerateRequest:
    return GenerateRequest(config_epoch=config_epoch, version=version, seed=1000 + version, lam=0.0855,
                           episodes=episodes, max_steps=40, res_token=f"tok-{version}")


def test_reconfigure_returns_incrementing_epoch(tmp_path):
    """Each successful configure advances the runner-assigned epoch; the client learns it from the
    reply (the source of the epoch it then sends in generate)."""
    with SubprocessActorTransport(_write_fake_runner(tmp_path)) as t:
        assert t.reconfigure(_cfg()) == 1
        assert t.reconfigure(_cfg()) == 2


def test_generate_returns_structured_meta(tmp_path):
    """A generate at the live epoch returns the structured meta (written + echoed epoch/version) — the
    typed replacement for the stderr `wrote N episode(s)` scrape."""
    with SubprocessActorTransport(_write_fake_runner(tmp_path)) as t:
        epoch = t.reconfigure(_cfg())
        res = t.generate(_req(epoch, version=42, episodes=300))
        assert isinstance(res, GenerateResult)
        assert res.written == 300 and res.config_epoch == epoch and res.version == 42


def test_new_version_same_epoch_not_rejected(tmp_path):
    """The common case — new trained weights, unchanged config — is a new-version / SAME-epoch generate,
    and it must NOT be rejected (the two gates are independent: version gates the weight reload, epoch
    gates config adoption)."""
    with SubprocessActorTransport(_write_fake_runner(tmp_path)) as t:
        epoch = t.reconfigure(_cfg())
        for v in (0, 1, 2, 3):
            res = t.generate(_req(epoch, version=v))
            assert res.config_epoch == epoch and res.version == v


def test_generate_epoch_mismatch_raises(tmp_path):
    """A generate carrying a stale epoch is a loud config_epoch_mismatch — the transport surfaces the
    runner's machine tag, never silently proceeds."""
    with SubprocessActorTransport(_write_fake_runner(tmp_path)) as t:
        t.reconfigure(_cfg())  # epoch 1
        with pytest.raises(ControlError) as ei:
            t.generate(_req(config_epoch=99))
        assert ei.value.tag == C.ERR_EPOCH_MISMATCH


def test_generate_before_configure_raises(tmp_path):
    """A generate before any configure is a loud not_configured (the runner has no env/policy yet)."""
    with SubprocessActorTransport(_write_fake_runner(tmp_path)) as t:
        with pytest.raises(ControlError) as ei:
            t.generate(_req(config_epoch=0))
        assert ei.value.tag == C.ERR_NOT_CONFIGURED


def test_instance_knob_change_rejected(tmp_path):
    """Changing an INSTANCE knob (instance_path) live is a loud instance_knob_changed reject — the env
    is built once; a change is a NEW experiment, not a live retune."""
    with SubprocessActorTransport(_write_fake_runner(tmp_path)) as t:
        assert t.reconfigure(_cfg(instance_path="A.json")) == 1
        with pytest.raises(ControlError) as ei:
            t.reconfigure(_cfg(instance_path="B.json"))
        assert ei.value.tag == C.ERR_INSTANCE_KNOB


def test_ping_serving_transitions_after_configure(tmp_path):
    """ping reports serving=False before the first configure (no env/policy) and True after — the
    readiness signal the executor's bounded-retry probe reads."""
    with SubprocessActorTransport(_write_fake_runner(tmp_path)) as t:
        before = t.ping()
        assert isinstance(before, PingResult) and before.serving is False and before.config_epoch == 0
        epoch = t.reconfigure(_cfg())
        after = t.ping()
        assert after.serving is True and after.config_epoch == epoch


def test_recv_timeout_reaps_and_raises(tmp_path):
    """A wedged runner (never replies to generate) trips the BOUNDED recv: a loud recv_timeout, and the
    process is reaped — the pipe analog of ZMQ_RCVTIMEO, the §2.4 non-hang net (not a forever-block)."""
    t = SubprocessActorTransport(_write_fake_runner(tmp_path), recv_timeout_s=1.0,
                                 extra_args=("--hang-generate",))
    try:
        epoch = t.reconfigure(_cfg())   # configure still replies (uses the ready timeout)
        with pytest.raises(ControlError) as ei:
            t.generate(_req(epoch))
        assert ei.value.tag == ERR_RECV_TIMEOUT
        assert t._proc.poll() is not None, "the wedged runner was not reaped on the recv timeout"
    finally:
        t.close()


def test_runner_died_raises(tmp_path):
    """A runner that exits without replying surfaces as a loud runner_died (stdout EOF), not a hang."""
    t = SubprocessActorTransport(_write_fake_runner(tmp_path), extra_args=("--die-on-generate",))
    try:
        epoch = t.reconfigure(_cfg())
        with pytest.raises(ControlError) as ei:
            t.generate(_req(epoch))
        assert ei.value.tag == ERR_RUNNER_DIED
    finally:
        t.close()


def test_close_is_idempotent(tmp_path):
    """close() is safe on every exit path and on repeat (graceful shutdown then bounded reap)."""
    t = SubprocessActorTransport(_write_fake_runner(tmp_path))
    t.reconfigure(_cfg())
    t.close()
    t.close()  # must not raise
    assert t._proc.poll() is not None


def test_spawn_failure_raises_loud():
    """A missing runner binary is a loud ControlError at construction (ADR-0002), not a silent no-op."""
    with pytest.raises(ControlError) as ei:
        SubprocessActorTransport("/nonexistent/chocofarm-cpp-runner-xyz")
    assert ei.value.tag == ERR_RUNNER_DIED
