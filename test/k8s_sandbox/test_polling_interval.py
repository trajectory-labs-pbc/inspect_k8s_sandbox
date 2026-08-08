from unittest.mock import MagicMock

import pytest
from pytest import MonkeyPatch

from k8s_sandbox._sandbox_environment import (
    DEFAULT_POLLING_INTERVAL,
    INSPECT_SANDBOX_POLLING_INTERVAL,
    K8sSandboxEnvironment,
)


def _sandbox() -> K8sSandboxEnvironment:
    # default_polling_interval reads only the environment, so the collaborators
    # can be inert: constructing a real Release/Pod would need a cluster.
    return K8sSandboxEnvironment(MagicMock(), MagicMock(), MagicMock())


def test_default_is_higher_than_inspects_base(monkeypatch: MonkeyPatch) -> None:
    """The whole point of the override: poll less often than Inspect's 2s.

    Asserted as a strict inequality against the base class rather than against a
    literal, so this fails if a future Inspect release raises its own default
    past ours and silently makes the override a no-op.
    """
    monkeypatch.delenv(INSPECT_SANDBOX_POLLING_INTERVAL, raising=False)
    from inspect_ai.util import SandboxEnvironment

    inspect_default = SandboxEnvironment.default_polling_interval(MagicMock())

    assert _sandbox().default_polling_interval() == DEFAULT_POLLING_INTERVAL
    assert DEFAULT_POLLING_INTERVAL > inspect_default


def test_env_var_overrides(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv(INSPECT_SANDBOX_POLLING_INTERVAL, "2.5")

    assert _sandbox().default_polling_interval() == 2.5


def test_env_var_accepts_integer_string(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv(INSPECT_SANDBOX_POLLING_INTERVAL, "30")

    assert _sandbox().default_polling_interval() == 30.0


def test_invalid_env_var_raises(monkeypatch: MonkeyPatch) -> None:
    """Fail loudly: a typo'd interval must not silently fall back to the default.

    Silently defaulting would reintroduce the exec-tax regression this override
    exists to remove, with nothing in the logs to explain the lost concurrency.
    """
    monkeypatch.setenv(INSPECT_SANDBOX_POLLING_INTERVAL, "not-a-number")

    with pytest.raises(ValueError, match=INSPECT_SANDBOX_POLLING_INTERVAL):
        _sandbox().default_polling_interval()
