# 🌊 PipelineSim — 液体管道瞬态模拟器

## 简介

PipelineSim 是一款基于 Python 的单相液体管道瞬态仿真工具，采用**特征线法（Method of Characteristics, MOC）**求解水击方程，耦合能量方程计算温度沿程分布。

对标商业软件 SPS（Stoner Pipeline Simulator）的单相瞬态分析能力。

## 核心能力

| 边界模式 | 入口条件 | 出口条件 | 计算结果 |
|---------|---------|---------|---------|
| Mode A | 流量 Q(t) + 温度 T(t) | 压力 P(t) | 沿线 P,T,Q 分布 |
| Mode B | 压力 P(t) + 温度 T(t) | 流量 Q(t) | 沿线 P,T,Q 分布 |

### 物理模型

**控制方程（一维瞬态流动）：**

连续方程：∂H/∂t + (a²/g)·∂V/∂x = 0

动量方程：∂H/∂x + (1/g)·∂V/∂t + f·V·|V|/(2gD) = 0

能量方程：ρ·cp·(∂T/∂t + V·∂T/∂x) = f·ρ·V³/(2D) - 4U·(T-T₀)/D

### 数值方法

- 水力学：特征线法（MOC），指定时间步长插值
- 温度场：迎风有限差分（Upwind FD），显式欧拉推进
- 物性耦合：密度 ρ(P,T)、黏度 μ(T) 每时间步更新

### 验证结果

**水击（Water Hammer）瞬时关闭测试：**

设定：L=10000m, D=0.5m, V₀=1.02m/s, 波速 a=1234.6m/s

| 指标 | 理论值 | 模拟值 | 偏差 |
|-----|-------|-------|------|
| 峰值压力 | 2.234 MPa | 2.213 MPa | 0.96% |
| 压力波周期 | 32.4s | — | — |

数值解与 Joukowsky 理论解高度吻合，偏差 < 1%。

## 快速开始

```python
from sim.fluid import Liquid
from sim.pipe import Pipe
from sim.solver import (
    SinglePhaseTransientSolver,
    flow_inlet, pressure_outlet
)

# 管道参数
pipe = Pipe(length=10000.0, diameter=0.5, wall_thickness=0.012)
liquid = Liquid()  # 水，默认参数

# 求解器
solver = SinglePhaseTransientSolver(pipe, liquid, Nx=50)

# Mode A: 入口流量+温度 / 出口压力
result = solver.solve(
    t_max=40.0,
    inlet_bc=flow_inlet(lambda t: 0.2, lambda t: 20.0),
    outlet_bc=pressure_outlet(lambda t: 1.0e6),
    mode='A',
    V_initial=1.02, T_initial=18.0,
)

# 结果：result.P (压力), result.T (温度), result.Q (流量)
```

## 项目结构

```
pipeline-sim/
├── sim/
│   ├── __init__.py
│   ├── fluid.py      # 液体物性（P,T依赖）
│   ├── pipe.py       # 管道几何 + 热传递
│   └── solver.py     # MOC + 温度场耦合求解器
├── examples/
│   ├── valve_closure.py        # 水击示例
│   ├── valve_stability_test.py # 阀门稳定性测试
│   └── test_both_modes.py      # 双模式测试
├── output/
│   └── *.png, *.json           # 结果输出
└── README.md
```

## 依赖

- Python 3.8+
- NumPy
- Matplotlib

## 后续开发方向

- [ ] 多管段串联/并联网络
- [ ] 气相段塞/两相流
- [ ] 稳态初始化计算
- [ ] 更丰富的边界条件（泵特性曲线、调节阀）

---

**🛸 Orbit | 2026-05-11**
