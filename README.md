# PipelineSim — 液体管道瞬态模拟

单相液体管道瞬态仿真工具，基于**特征线法 (Method of Characteristics, MOC)**，用于模拟水击/压力波动等瞬态过程。

## 与商业软件对标

| 软件 | 定位 | 本项目的对标目标 |
|------|------|-----------------|
| SPS (Stoner) | 液体管道动态仿真 | ✅ 单相瞬态求解器 |
| OLGA | 多相流动态仿真 | 🚧 远期目标 |
| LedaFlow | 多相流瞬态仿真 | 🚧 远期目标 |

## 物理模型

### 控制方程（一维瞬态流动）

连续方程：
```
∂H/∂t + (a²/g) · ∂V/∂x = 0
```

动量方程：
```
∂H/∂x + (1/g) · ∂V/∂t + f·V·|V|/(2gD) = 0
```

其中：
- H = 压力水头 (m)
- V = 流速 (m/s)
- a = 水击波速 (m/s)
- f = Darcy-Weisbach 摩阻系数
- D = 管道内径 (m)
- g = 重力加速度 (9.81 m/s²)

### 特征线法 (MOC)

PDE 沿特征线方向转化为常微分方程：

C⁺: dH/dt + (a/g)·dV/dt + (a·f·V·|V|)/(2gD) = 0, 沿 dx/dt = a

C⁻: dH/dt - (a/g)·dV/dt - (a·f·V·|V|)/(2gD) = 0, 沿 dx/dt = -a

## 项目结构

```
pipeline-sim/
├── sim/
│   ├── __init__.py
│   ├── physics.py      # 物理参数（波速、摩阻）
│   ├── solver.py       # MOC 求解器
│   ├── boundary.py     # 边界条件（储罐、阀门、泵）
│   └── network.py      # 管道网络拓扑
├── examples/
│   ├── simple_pipe.py  # 简单单管示例
│   └── valve_closure.py# 阀门关闭水击
├── tests/
│   └── test_moc.py
├── output/             # 模拟结果
├── setup.py
└── README.md
```

## 快速开始

```bash
cd examples
python valve_closure.py
```

## 依赖

- Python 3.8+
- NumPy
- Matplotlib (可视化)
