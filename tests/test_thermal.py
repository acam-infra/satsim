"""ThermalModel：explicit 与旧逻辑一致；dynamic 升温后节流。"""
from __future__ import annotations

from satsim.input.schema import SpecError, _validate_thermal
from satsim.method.engine.timing import clock_factor as dvfs_clock_factor
from satsim.method.policies.thermal.dynamic import DynamicThermal
from satsim.method.policies.thermal.explicit import ExplicitThermal
from satsim.method.policies.thermal import thermal_from_cfg


def _explicit_cfg(**extra):
    th = dict(P_idle=60, P_max=300, gamma=2.4, f_min=0.35, **extra)
    return th


def test_explicit_matches_legacy_inline():
    th = _explicit_cfg(
        explicit_events=[[2, 0.0, 40.0, 80.0], [0, 40.0, 1.0e9, 80.0]],
    )
    model = ExplicitThermal(th)

    def legacy(node: int, t: float) -> float:
        p = th["P_max"]
        for ev in th["explicit_events"]:
            j, t0, t1, pc = ev
            if j == node and t0 <= t < t1:
                p = min(p, pc)
        return dvfs_clock_factor(p, th)

    for node in (0, 1, 2, 3):
        for t in (0.0, 20.0, 40.0, 100.0):
            assert model.clock_factor(node, t) == legacy(node, t)


def test_thermal_from_cfg_default_explicit():
    cfg = {"nodes": 4, "thermal": _explicit_cfg()}
    m = thermal_from_cfg(cfg)
    assert m.mode == "explicit"
    assert m.clock_factor(0, 0.0) == 1.0


def test_dynamic_heats_and_throttles():
    th = _explicit_cfg(
        mode="dynamic",
        tau_s=10.0,
        R_th=0.5,  # °C/W；忙时稳态 ~ T_amb + 150 = 170，很快过 84
        T_amb_C=20.0,
        T0_C=40.0,
        T_throttle_C=84.0,  # LAB-1178/1214 叙事
        T_release_C=78.0,
        f_throttle=0.71,
        P_busy=300.0,
    )
    m = DynamicThermal(th, n=2)
    assert m.clock_factor(0, 0.0) == 1.0
    # 节点 0 有负载，推一段时间应越过节流点
    changed = False
    t = 0.0
    for _ in range(50):
        t += 1.0
        if m.advance(t, [1, 0]):
            changed = True
            break
    assert changed
    assert m.temperature(0) >= 84.0
    assert m.clock_factor(0, t) == 0.71
    # 空闲节点不应被节流
    assert m.clock_factor(1, t) == 1.0


def test_dynamic_release_on_cool_down():
    th = _explicit_cfg(
        mode="dynamic",
        tau_s=5.0,
        R_th=0.5,
        T_amb_C=20.0,
        T0_C=90.0,  # 已过热
        T_throttle_C=84.0,
        T_release_C=78.0,
        f_throttle=0.71,
        P_busy=300.0,
    )
    m = DynamicThermal(th, n=1)
    m.advance(0.0, [1])
    # 强制走一遍滞回进入节流
    assert m.advance(0.01, [1]) or m.clock_factor(0, 0.01) == 0.71
    # 空载冷却到释放点以下
    t = 0.01
    released = False
    for _ in range(200):
        t += 1.0
        if m.advance(t, [0]):
            if m.clock_factor(0, t) == 1.0:
                released = True
                break
    assert released
    assert m.temperature(0) <= 78.0


def test_dynamic_schema_rejects_bad_hysteresis():
    th = _explicit_cfg(
        mode="dynamic",
        tau_s=10.0,
        R_th=0.4,
        T_amb_C=20.0,
        T0_C=40.0,
        T_throttle_C=84.0,
        T_release_C=90.0,  # bad
        f_throttle=0.71,
        P_busy=300.0,
    )
    try:
        _validate_thermal(th)
        raise AssertionError("expected SpecError")
    except SpecError as e:
        assert "T_release_C" in str(e)
