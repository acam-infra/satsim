# SPEC：可插拔算法框架（input / method / output）

状态：已落地（仅显式策略；无旧拼盘兼容层）  
范围：本地重构 `/workspace/satsim`；不改 C1/C2 语义；不推远程。

## 1. 目标与非目标

### 目标
- 顶层科研抽象固定为三层：**输入 → 方法 → 输出**。外人 5 分钟能对上论文实验。
- **方法**内部再拆：`engine`（通用内核）+ `interfaces`（窄接口）+ `policies`（baseline / 对比实现）。
- **数据生成**属于输入，默认**线下**：仿真开始前生成完整请求序列，引擎只回放。
- YAML 用 **policy id + 旋钮** 选择实现组合；`*_nobg` 等是配置旋钮，不是独立算法。
- 新增对比方法 = 新文件 + registry 一行 + YAML 改 id，不改引擎主循环。
- 现有 scenario（`stac_s1_cost_claims_v3`）与 C1/C2 判定**行为不变**。

### 非目标
- 不重写物理/轨道模型，不扩成通用星座 OS。
- 不改 C1/C2 定义与达标口径（换位后 TPOT、goodput 等）。
- 只认显式 `placement` / `rebalance` / `link_cost` + `knobs`；无 route/migrate/cfg 双轨。
- `Request` 已中立化（`pin`/`pout` 仍为别名）。不推远程。

## 2. 目录树与模块职责

```text
satsim/
├── README.md
├── GLOSSARY.md
├── AGENTS.md
├── configs/                         # 可跑 YAML 入口（自 input/scenarios 迁或兼容）
│   ├── construct_s1.yaml
│   └── s1_cost_claims_v3.yaml       # 可与旧名并存
├── input/                           # 兼容：场景 YAML + 叙事 scene（旧路径仍可用）
│   ├── scenarios/
│   └── scenes/
├── docs/
│   └── SPEC_PLUGIN_FRAMEWORK.md     # 本文件
├── src/satsim/
│   ├── __main__.py                  # CLI：跑一份 spec → claims
│   ├── input/                       # 输入层
│   │   ├── load.py                  # YAML 加载、resolve、job 展开
│   │   ├── schema.py                # 准入表、校验、旋钮理由
│   │   └── generate/                # 线下数据生成（可插拔）
│   │       ├── interfaces.py        # ArrivalProcess / WorkloadGenerator
│   │       ├── synthetic_pass.py    # 现网：压缩过境（已有）
│   │       └── poisson.py           # 泊松到达（已准入）
│   ├── method/                      # 方法层
│   │   ├── engine/                  # 通用内核，不含具体算法名
│   │   │   ├── world.py             # 节点/请求/链路世界状态
│   │   │   ├── loop.py              # SimPy 进程与 run_one
│   │   │   └── net.py               # 链路带宽分配（可与 world 同文件起步）
│   │   ├── interfaces/              # 窄接口
│   │   │   ├── placement.py
│   │   │   ├── rebalance.py
│   │   │   └── link_cost.py
│   │   ├── policies/                # baseline / 对比方法
│   │   │   ├── placement/
│   │   │   │   └── first.py
│   │   │   ├── rebalance/
│   │   │   │   ├── none.py
│   │   │   │   ├── local.py         # 换位后重放，KV 不上链路（cost free）
│   │   │   │   └── with_link.py     # 整段 KV 上链路
│   │   │   └── link_cost/
│   │   │       ├── off.py           # 原 free：瞬时搬走
│   │   │       ├── full.py          # 原 full：整段 KV 走剩余带宽
│   │   │       └── net.py           # 原 net：核对代价是否来自链路
│   │   └── registry.py              # policy id → 接口组合 + 旋钮
│   └── output/                      # 输出层：只读 run rows
│       ├── metrics.py               # 单次 run 行指标（可薄封装）
│       └── claims.py                # C1/C2 汇总（语义不变）
└── tests/
    ├── test_claims.py
    └── golden/                      # pytest 自动对比摘录
```

### 职责边界
| 层 | 做什么 | 不做什么 |
|---|---|---|
| `input` | 载入 YAML、校验准入、线下生成请求序列、展开 (policy×seed) jobs | 放置、迁移、链路调度 |
| `method.engine` | 推进世界：到达回放、解码、功率帽换位、背景流、调用接口 | 知道 `first` / `rebalance_link` 等策略名 |
| `method.interfaces` | 定义 Placement / Rebalance / LinkCost 契约 | 含算法逻辑 |
| `method.policies` | 实现具体 baseline / 对比方法 | 改 SimPy 主循环结构 |
| `output` | 读 rows → 指标与 C1/C2 | 改世界状态或再跑仿真 |

## 3. 三个接口（草案签名）

类型名可用 `Protocol`；世界对象以最小只读/可变面暴露（首轮可传现有 `World`）。

```python
# method/interfaces/placement.py
class PlacementPolicy(Protocol):
    def place(self, world: WorldView, req: Request) -> int:
        """首次放置：返回节点 id。放不下应抛错（与现 admit 一致）。"""
```

```python
# method/interfaces/rebalance.py
class RebalancePolicy(Protocol):
    def maybe_migrate(self, world: WorldView) -> None:
        """控制周期触发：决定是否发起一次迁移（可 env.process）。
        none 实现为空操作。"""
```

```python
# method/interfaces/link_cost.py
class LinkCostModel(Protocol):
    mode: str  # "free" | "full" | "net" —— 与现 cost.mode 对齐，避免改 claim

    def transfer(
        self,
        world: WorldView,
        req: Request,
        src: int,
        dst: int,
        nbytes: float,
    ) -> Generator:
        """迁移时如何付链路代价。free：timeout(0)；full/net：走 Net 剩余带宽。
        返回 SimPy generator（由 engine 以 process 跑）。"""
```

放置/重分配/链路代价逻辑在 `policies/`；engine 只调接口。

## 4. input.generate 线下流程

```text
YAML.trace + workload + seed
        │
        ▼
  ArrivalProcess.generate(...)   # 纯函数/无 SimPy
        │
        ▼
  list[Request] + anchor 元数据   # 完整序列
        │
        ▼
  method.engine 按 req.t 回放 decode
```

- **默认离线**：一次 `generate` 产出整窗请求，再 `env.run`。
- 现网生成器：`synthetic_pass`（由 `make_pass` 迁入 `generate/synthetic_pass.py`）。
- 预留：`poisson.py` 实现同一 `ArrivalProcess` 接口（参数 λ、窗长等）；未准入 YAML `kind` 前不进主实验。
- 生成器**只**产出到达时刻、长短、token pad 等；不碰放置/迁移。

```python
class ArrivalProcess(Protocol):
    def generate(
        self,
        trace: dict,
        *,
        p_long: float,
        seed: int,
    ) -> tuple[list[Request], float, int]:
        """返回 (请求列表, max_cluster 锚点时刻, 簇计数)。"""
```

## 5. YAML：policy id + 旋钮 → 接口组合

### 5.1 实验臂（显式）

| 臂名 | placement | rebalance | link_cost | knobs |
|---|---|---|---|---|
| `place_once` | first | none | off | — |
| `rebalance_local` | first | local | off | — |
| `rebalance_over_link` | first | with_link | full | — |
| `rebalance_over_link_net` | first | with_link | net | — |
| `rebalance_local_nobg` | first | local | off | background=0 |
| `rebalance_over_link_nobg` | first | with_link | full | background=0 |

### 5.2 YAML 形态



```yaml
policies:
  rebalance_link:
    placement: first
    rebalance: with_link
    link_cost: full
    hysteresis: 0.15
    knobs:
      background.gbps_per_node: 2.75   # 或 0 表示 nobg
```

`registry.resolve(name, pol_yaml) -> PolicyBundle(placement, rebalance, link_cost, cfg_overrides)`。

## 6. output/claims 与 engine 的边界

- `engine.run_one`（或 `loop.run_one`）产出**一行结构化 dict**（status、good_long、mig、…），**不**解释 C1/C2。
- `output/metrics.py`：可选，规范化单行字段名。
- `output/claims.py`：只读多行 + evidence → C1/C2；继续按策略**显示名**（`first`、`rebalance`、…）聚合，避免改测试断言。
- 引擎不得 `import claims`；claims 不得 `import engine` 并开跑。

## 7. 迁移阶段（P0–P5 精简）

| 阶段 | 内容 | 验收 |
|---|---|---|
| **P0** | 冻结：现有 pytest 绿；可选写入 `tests/golden/` 摘要 | `uv run pytest` 通过 |
| **P1** | 建目录骨架；抽出 interfaces；engine 调 interfaces；逻辑在 policies | 行为不变 |
| **P2** | `make_pass` → `input/generate/synthetic_pass`；加 `poisson` 骨架 | 生成器可跑 |
| **P3** | `registry`：六名字 → 接口组合；policies 分文件 | 显式 YAML 可跑 |
| **P4** | `output/metrics` 薄层；claims 仍只读 rows | C1/C2 断言不变 |
| **P5** | `configs/` 入口兼容；文档/README 指向本 SPEC；命名收口可另开 | 旧 `input/scenarios/...` 仍可用 |

本轮实现：**P0–P4 骨架 + 逻辑迁入 + 兼容层**；P5 部分（configs 兼容链接）。

## 8. 验收标准

1. `uv run pytest` 全部通过（含 `test_benefit_and_link_cost`）。
2. `uv run satsim input/scenarios/stac_s1_cost_claims_v3.yaml` 可跑出 C1/C2。
3. 源码树出现 `method/engine`、`method/interfaces`、`method/policies`、`input/generate`。
4. 引擎主循环不出现对具体策略名字符串的分支（允许 registry 在加载期解析）。
5. 不推远程、不开 PR。

## 9. 与现有代码对应关系

| 现有 | 迁入后 |
|---|---|
| 原 monolithic `spec`/`sim` | `input/schema.py` + `input/load.py`；`method/engine/`；**无** re-export 壳 |
| `Req`/`Net`/`World` | `domain.Request` + `method/engine/world.py` |
| decode/swap/control/`run_one` | `method/engine/loop.py` |
| 到达生成 | `input/generate/`（`generate_arrivals`） |
| `World.admit` | `policies/placement/first.py` |
| `World.try_move` + migrate（free） | `policies/rebalance/local.py` + `link_cost/off.py` |
| migrate full/net | `policies/rebalance/with_link.py` + `link_cost/full.py`/`net.py` |
| `output/claims.py` | 保留；可选 `metrics.py` |
| `input/scenarios/*.yaml` | 保留；`configs/` 可 symlink 或复制入口 |
| `__main__.py` | 改为走 load + generate + engine + claims |

## 10. 实现约定（本轮代码）

1. **先 SPEC、后代码**（本文件为 SPEC）。
2. 导入走 `input.load` / `input.schema` / `method.engine`；已删除 `spec.py` / `sim.py` 兼容壳。
3. C1/C2 按行内 placement/rebalance/link_cost/nobg 选臂。
4. `poisson` 已准入。



## 11. 落地备忘（无兼容层）

- 主场景策略名：`place_once` / `rebalance_local` / `rebalance_over_link` / `rebalance_over_link_net` / `*_nobg`。
- `cost.mode` 由 `link_cost` 推导写入 cfg，禁止手写进 knobs。
- 已删除：`src/satsim/input/spec.py`、`src/satsim/method/sim.py`、LEGACY 映射、旧拼盘校验。
- Golden：`tests/test_golden.py`。
