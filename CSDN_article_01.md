# 🚀 PipelineSim：从零实现一个工业级管道瞬态模拟器（开源Python版）

> **如果你做油气管道、供水管网或化工流体仿真，这篇文章就是为你写的。**

## 为什么要有PipelineSim？

在石油天然气行业，管道瞬态仿真的标准工具是 **SPS（Stoner Pipeline Simulator）** 和 **OLGA**，年授权费动辄 **$20万+**。开源生态里呢？

- 大多是 **MATLAB** 项目（auralius/waterhammer）
- 或者 **C++** 硬核实现（FSund/transient-pipeline-flow）
- **Python生态几乎是空白**

这就是 **PipelineSim** 诞生的原因 —— 一个纯Python实现的、对标工业级单相液体管道瞬态模拟器。

## 核心能力：两种边界模式

| 模式 | 入口条件 | 出口条件 | 能算什么 |
|------|---------|---------|---------|
| **Mode A** | 流量 Q(t) + 温度 T(t) | 压力 P(t) | 全线压力/温度/流量 |
| **Mode B** | 压力 P(t) + 温度 T(t) | 流量 Q(t) | 全线压力/温度/流量 |

## 四大数值求解器

PipelineSim 一口气实现了4种求解器，这在开源界是独一份：

### 1️⃣ MOC — 特征线法（基准）
经典的指定时间步长法，CFL=1.0。水锤验证与Joukowsky理论偏差 < 1%。

**适用场景**：精确水击分析、阀门快速关闭

### 2️⃣ IFDM — 隐式有限差分
Crank-Nicolson时间推进，无条件稳定，允许大时间步长。

**适用场景**：慢速瞬变、长时段仿真（数小时到数天）

### 3️⃣ MacCormack — 预测-校正法
二阶精度，能捕捉陡峭压力波前。

**适用场景**：需要高精度波前捕捉的快瞬变

### 4️⃣ FVM — 有限体积法（MUSCL-Hancock）
守恒格式，强激波捕捉能力，二阶重构+Minmod限制器。

**适用场景**：存在大压力梯度的极端工况

### 求解器对比实测

测试场景：L=10km，D=0.5m，Q₀=0.2m³/s，2秒内阀门关闭

| 求解器 | 峰值压力 | 偏差 | 时间步数 |
|--------|---------|------|---------|
| MOC | 2.071 MPa | 基准 | 112步 |
| IFDM | 1.998 MPa | -3.5% | 75步(1.5×CFL) |
| MacCormack | 2.141 MPa | +3.4% | 139步 |
| FVM | 2.070 MPa | -0.05% | 139步 |

**结论**：四种求解器均收敛到Joukowsky理论值附近，MOC和FVM精度最高，IFDM在大时间步长下略有扩散。

## 快速上手

```bash
# 安装
git clone https://github.com/chendw79/pipeline-sim.git
cd pipeline-sim
pip install numpy matplotlib

# 稳态分析
python -m sim.cli analyze --length 15000 --diameter 0.6 --flow 0.25

# 瞬态仿真（阀门关闭水击）
python -m sim.cli water-hammer --length 10000 --flow 0.2 --closure-time 5

# 求解器对比
python -c "
from sim.fluid import Liquid
from sim.pipe import Pipe
from sim.solver import flow_inlet, pressure_outlet
from sim.solver_advanced import compare_solvers

pipe = Pipe(10000, 0.5, 0.012)
liquid = Liquid()

# 阀门2秒关闭
results = compare_solvers(pipe, liquid,
    inlet_bc=flow_inlet(lambda t: 0.2 if t<2 else 0, lambda t: 20),
    outlet_bc=pressure_outlet(lambda t: 1e6),
    mode='A', Nx=30, t_max=30)

for name, r in results.items():
    print(f'{name}: Pmax={r.P.max()/1e6:.2f} MPa')
"
```

## 路线图

```
Phase A: 核心求解器 ── ✅ 已完成
Phase B: 专业功能 ── 🔄 进行中（阀门模型/文档/可视化）
Phase C: 差异化 ── 📋 规划中（泄漏检测/PyPI发布/实时面板）
```

## 开源地址

GitHub: [github.com/chendw79/pipeline-sim](https://github.com/chendw79/pipeline-sim)

---

**作者：** ChenDavid
**技术栈：** Python, NumPy, Matplotlib, Plotly
**许可证：** MIT
