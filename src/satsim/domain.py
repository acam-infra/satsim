"""中立领域类型：输入生成与方法引擎共用，避免 input ↔ method 互相拖类型。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Request:
    """一次推理请求（到达记录 + 运行期驻留字段）。"""

    i: int
    t: float
    long: bool
    prompt_tokens: int  # 输入 token（原 pin）
    output_tokens: int  # 输出 token（原 pout）
    node: int = -1
    gen: float = 0.0
    t_first: float | None = None
    t_done: float | None = None
    migrating: bool = False
    n_migrations: int = 0  # 原 nmig
    proc: object | None = None
    post_tok: float = 0.0
    post_time: float = 0.0

    # 兼容旧字段名（引擎/策略大量使用 pin/pout/nmig）
    @property
    def pin(self) -> int:
        return self.prompt_tokens

    @pin.setter
    def pin(self, value: int) -> None:
        self.prompt_tokens = int(value)

    @property
    def pout(self) -> int:
        return self.output_tokens

    @pout.setter
    def pout(self, value: int) -> None:
        self.output_tokens = int(value)

    @property
    def nmig(self) -> int:
        return self.n_migrations

    @nmig.setter
    def nmig(self, value: int) -> None:
        self.n_migrations = int(value)

    @property
    def kv(self) -> float:
        return self.prompt_tokens + self.gen


# 历史别名
Req = Request
