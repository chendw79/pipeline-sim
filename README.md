# 🌊 PipelineSim

**从零实现的液体管道瞬态模拟器 — 对标 SPS/OLGA 单相瞬态能力**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-9/9-passing-brightgreen)]()
[![GitHub](https://img.shields.io/badge/GitHub-chendw79%2Fpipeline--sim-blue)](https://github.com/chendw79/pipeline-sim)

---

**PipelineSim** 是一款基于 Python 的单相液体管道瞬态仿真工具，采用**特征线法（Method of Characteristics, MOC）**求解水击方程，耦合能量方程计算温度沿程分布。

## Why PipelineSim?

> 开源界缺少 Python 语言的液体管道瞬态模拟器。现有项目或是 MATLAB (auralius/waterhammer)、C++ (FSund/transient-pipeline-flow)，或者只是数据分析项目。**PipelineSim 填补了这个空白。**

## 核心能力

| 边界模式 | 入口条件 | 出口条件 | 计算结果 |
|---------|---------|---------|---------|
| **Mode A** | 流量 Q(t) + 温度 T(t) | 压力 P(t) | 沿线 P,T,Q 分布 |
| **Mode B** | 压力 P(t) + 温度 T(t) | 流量 Q(t) | 沿线 P,T,Q 分布 |

### 数值方法

- **水力学**: 特征线法 (MOC)，指定时间步长，CFL = 1.0
- **温度场**: 迎风有限差分 (Upwind FD)，显式欧拉推进
- **物性耦合**: 密度 ρ(P,T)、黏度 μ(T) 每时间步更新
- **稳态初始化**: 分析式 P/T 剖面，消除启动瞬态误差

## 快速开始

```bash
git clone https://github.com/chendw79/pipeline-sim.git
cd pipeline-sim
python -m venv venv && source venv/bin/activate
pip install numpy matplotlib
```

```python
from sim.fluid import Liquid
from sim.pipe import Pipe
from sim.steady import SteadyStateCalculator
from sim.solver import SinglePhaseTransientSolver, flow_inlet, pressure_outlet

# 管道参数
pipe = Pipe(length=15000.0, diameter=0.6, wall_thickness=0.014)
liquid = Liquid(name="Crude Oil", rho_ref=860.0, bulk_modulus=1.8e9, cp=2000.0)

# Step 1: 稳态初始化（关键！）
calc = SteadyStateCalculator(pipe, liquid)
V0, P_init, T_init = calc.initialize_transient(Q=0.25, T_inlet=45.0, P_outlet=0.5e6)

# Step 2: 瞬态模拟
solver = SinglePhaseTransientSolver(pipe, liquid, Nx=40)
result = solver.solve(
    t_max=80.0,
    inlet_bc=flow_inlet(lambda t: 0.25, lambda t: 45.0),
    outlet_bc=pressure_outlet(lambda t: 0.5e6),
    mode='A',
    V_initial=V0,
    P_initial=P_init,
    T_initial=T_init,
)

# Step 3: 导出结果
from sim.export import export_to_csv, generate_report
export_to_csv(result, 'results.csv', pipe)
print(generate_report(result, pipe, liquid, solver, "My Pipeline"))
```

输出示例:
```
📊 Pipeline Analysis Report
  Fluid: Crude Oil
  Pipe: L=15.0km, D=600mm
  Flow: 250 L/s, V=0.88 m/s
  Re=38020, f=0.0224
  P_inlet=1.110 MPa, P_outlet=0.500 MPa
  T_inlet=45.0°C, T_outlet=28.2°C (drop 16.8°C)
  Thermal time scale: L/V = 4.7 hours
```

## 功能矩阵

| 模块 | 功能 | 状态 |
|------|------|------|
| `sim/solver.py` | MOC 水力学 (Mode A/B, 双模式) | ✅ |
| `sim/solver.py` | 耦合温度场 (摩擦生热+散热) | ✅ |
| `sim/fluid.py` | 液体物性 (P/T依赖) | ✅ |
| `sim/pipe.py` | 管道几何 + 波速计算 | ✅ |
| `sim/steady.py` | **稳态初始化** (分析式 P/T 剖面) | ✅ NEW |
| `sim/network.py` | **多管段串联网络** | ✅ NEW |
| `sim/pump.py` | **离心泵模型** (相似定律) | ✅ NEW |
| `sim/export.py` | **CSV/JSON/HDF5 导出** | ✅ NEW |
| `sim/validation.py` | **输入验证** | ✅ NEW |
| `sim/solver.py` | 泵入口边界条件 | ✅ NEW |

## 验证结果

**水击 (Water Hammer) 瞬时关闭测试:**
- 管道: L=10km, D=0.5m, 钢管
- 流体: 水 (ρ=998 kg/m³, K=2.15 GPa)
- 波速: a=1234.6 m/s
- Joukowsky 理论峰值: 2.234 MPa
- 模拟峰值: 2.213 MPa
- **偏差: 0.96%** 🎯

## 工程洞察

> **热力时间尺度 >> 水力时间尺度**
>
> 对 15km 原油管线: 热力平衡需 ~4.7 小时，水力平衡仅需 ~10 秒。
> 这是为什么稳态初始化如此关键——没有它，启动瞬态会污染结果。

## 项目结构

```
pipeline-sim/
├── sim/          # 核心模块 (9 files)
│   ├── solver.py       # MOC 求解器 🎯
│   ├── steady.py       # 稳态初始化
│   ├── network.py      # 多管段网络
│   ├── pump.py         # 离心泵
│   ├── export.py       # 数据导出
│   ├── validation.py   # 输入验证
│   ├── fluid.py        # 流体物性
│   └── pipe.py         # 管道参数
├── examples/     # 示例 (6 files)
│   ├── test_suite.py            # 9项测试
│   ├── professional_workflow.py # 专业工作流
│   ├── comprehensive_analysis.py# 综合分析
│   ├── valve_closure.py         # 水击测试
│   ├── valve_stability_test.py  # 稳定性测试
│   └── test_both_modes.py       # 双模式测试
├── docs/         # 文档
│   ├── README_CN.md     # 中文文档
│   └── article_csdn.md  # 技术文章
└── output/       # 生成结果
```

## 开发路线

```
Phase A: Core (当前)       → Phase B: Pro        → Phase C: Differentiator
─────────────────────────────────────────────────────────────────────
✅ MOC求解器                 📋 控制阀模型          📋 最优控制
✅ 温度场                    📋 瞬态文件输入         📋 PINN校准
✅ 稳态初始化                📋 命令行接口           📋 不确定性量化
✅ 多管段网络 (NEW)          📋 HDF5导出            📋 泄露检测
✅ 泵模型 (NEW)              📋 API完善             📋 实时仪表盘
✅ 测试套件 (NEW)      
```

## 测试

```bash
python examples/test_suite.py

============================================================
PipelineSim Test Suite
============================================================
Test 1: Fluid Properties  ✅
Test 2: Pipe Properties   ✅
Test 3: Steady-State      ✅
Test 4: Solver (MOC)      ✅
Test 5: BCs               ✅
Test 6: Series Network    ✅
Test 7: Pump Model        ✅
Test 8: Validation        ✅
Test 9: Export            ✅
============================================================
Results: 9/9 passed ✅
```

## 依赖

- Python ≥ 3.8
- NumPy
- Matplotlib (仅用于绘图)

## 技术文章

详见 [`docs/article_csdn.md`](docs/article_csdn.md)

## 调研报告

详见 [`pipeline-sim-research`](https://github.com/chendw79/pipeline-sim-research)

## License

MIT

---

**🛸 Orbit | Built from first principles**
