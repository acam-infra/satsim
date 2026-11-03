"""每节点一阶 RC：温度越限则两档节流（示意，非完整轨道热平衡）。

默认阈值来自既有 lab 叙事（LAB-1178/1214）：约 84°C 触发、地板约 0.71×；
不是把外部 A100 标定直接当本仓物理真理，YAML 可改。
"""
from __future__ import annotations

import math
from typing import Sequence


class DynamicThermal:
    mode = "dynamic"

    def __init__(self, thermal: dict, *, n: int):
        self._th = thermal
        self.n = n
        self.tau_s = float(thermal["tau_s"])
        self.R_th = float(thermal["R_th"])
        self.T_amb_C = float(thermal["T_amb_C"])
        self.T_throttle_C = float(thermal["T_throttle_C"])
        self.T_release_C = float(thermal["T_release_C"])
        self.f_throttle = float(thermal["f_throttle"])
        self.P_busy = float(thermal["P_busy"])
        self.P_idle = float(thermal["P_idle"])
        self.P_max = float(thermal["P_max"])
        self.gamma = float(thermal["gamma"])
        self.f_min = float(thermal["f_min"])
        T0 = float(thermal["T0_C"])
        self._T = [T0] * n
        self._throttled = [T0 >= float(thermal["T_throttle_C"])] * n
        self._f = [1.0] * n
        self._t = 0.0
        self._loads = [0] * n
        self._peak_T = list(self._T)
        self._throttle_time_s = [0.0] * n
        self._sync_f()

    def _power(self, j: int) -> float:
        if self._loads[j] <= 0:
            return self.P_idle
        if self._throttled[j]:
            # 节流后发热与 DVFS 地板一致：P = P_idle + (P_max-P_idle) * f^gamma
            return self.P_idle + (self.P_max - self.P_idle) * (self.f_throttle**self.gamma)
        return self.P_busy

    def _sync_f(self) -> None:
        floor = max(self.f_throttle, self.f_min)
        for j in range(self.n):
            self._f[j] = floor if self._throttled[j] else 1.0

    def _apply_hysteresis(self) -> bool:
        changed = False
        for j in range(self.n):
            was = self._throttled[j]
            if self._throttled[j]:
                if self._T[j] <= self.T_release_C:
                    self._throttled[j] = False
            else:
                if self._T[j] >= self.T_throttle_C:
                    self._throttled[j] = True
            if self._throttled[j] != was:
                changed = True
        if changed:
            self._sync_f()
        return changed

    def _step(self, dt: float) -> bool:
        if dt <= 0:
            return False
        # 用步进前的节流态算功率，再积分温度，最后刷新滞回
        for j in range(self.n):
            if self._throttled[j]:
                self._throttle_time_s[j] += dt
            p = self._power(j)
            t_eq = self.T_amb_C + self.R_th * p
            decay = math.exp(-dt / self.tau_s)
            self._T[j] = t_eq + (self._T[j] - t_eq) * decay
            if self._T[j] > self._peak_T[j]:
                self._peak_T[j] = self._T[j]
        return self._apply_hysteresis()

    def advance(self, t: float, node_loads: Sequence[int]) -> bool:
        if len(node_loads) != self.n:
            raise ValueError(f"node_loads 长度 {len(node_loads)} != nodes {self.n}")
        self._loads = [int(x) for x in node_loads]
        dt = float(t) - self._t
        self._t = float(t)
        return self._step(dt)

    def clock_factor(self, node: int, t: float) -> float:
        # 控制周期之间懒推进一步（载荷沿用上次 advance）
        dt = float(t) - self._t
        if dt > 0:
            self._t = float(t)
            self._step(dt)
        return self._f[node]

    def temperature(self, node: int) -> float:
        """测试与诊断用。"""
        return self._T[node]

    def diagnostics(self) -> dict:
        """峰值温度与累计节流时间（秒），供实验小结。"""
        return {
            "peak_T_C": list(self._peak_T),
            "peak_T_C_max": max(self._peak_T) if self._peak_T else None,
            "throttle_time_s": list(self._throttle_time_s),
            "throttle_time_s_sum": float(sum(self._throttle_time_s)),
            "nodes_ever_throttled": int(sum(1 for x in self._throttle_time_s if x > 0)),
        }
