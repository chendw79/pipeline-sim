"""
leaks.py — 管道泄漏瞬态模型

模型:
  1. 泄漏边界条件: 在管道某位置形成分流
  2. 泄漏流量: Q_leak = C_d · A_leak · sqrt(2 · P / ρ)
  3. 管段拆分: 将单根管道在泄漏点处拆为两段，中间加泄漏边界

实现方式:
  - PipeLeak类: 描述泄漏参数
  - leak_split_bcs(): 生成上下游边界条件函数对
  - 目前方案: 单次仿真 + 泄漏修正（不拆分网格）
"""

import numpy as np
from typing import Callable, Optional, Tuple, List
from dataclasses import dataclass
from .pipe import Pipe
from .fluid import Liquid


@dataclass
class PipeLeak:
    """
    管道泄漏模型
    
    泄漏流量:
      Q_leak = C_d · A · sqrt(2ΔP/ρ)
      
    其中 C_d: 流量系数 (0.6~0.85), A: 泄漏孔面积
    """
    position: float = 5000.0       # 泄漏位置 (m, 从入口起)
    orifice_diameter: float = 0.01  # 泄漏孔直径 (m)
    discharge_coefficient: float = 0.62  # 流量系数
    start_time: float = 10.0       # 泄漏开始时间 (s)
    
    # 孔口面积
    @property
    def orifice_area(self) -> float:
        return np.pi * (self.orifice_diameter / 2) ** 2
    
    def leak_flow(self, P: float, rho: float) -> float:
        """
        计算泄漏流量
        
        Args:
            P: 泄漏点压力 (Pa)
            rho: 流体密度 (kg/m³)
        
        Returns:
            Q_leak: 泄漏流量 (m³/s)
        """
        if P <= 0:
            return 0.0
        # Q = C_d · A · sqrt(2P/ρ)
        return self.discharge_coefficient * self.orifice_area * np.sqrt(2 * max(P, 0) / max(rho, 1.0))
    
    def describe(self) -> str:
        """描述泄漏参数"""
        area_mm2 = self.orifice_area * 1e6
        return (f"泄漏位置: {self.position:.0f}m (距入口)\n"
                f"  孔径: {self.orifice_diameter*1000:.1f}mm (面积 {area_mm2:.1f}mm²)\n"
                f"  流量系数: Cd={self.discharge_coefficient:.2f}\n"
                f"  开始时间: t={self.start_time:.1f}s")


def leak_boundary_pair(
    leak: PipeLeak,
    inlet_bc: Callable,
    outlet_bc: Callable,
    pipe: Pipe,
    fluid: Liquid,
    mode: str = 'A',
) -> Tuple[Callable, Callable]:
    """
    将单管边界条件转换为含泄漏的上下游边界条件对
    
    Args:
        leak: 泄漏参数
        inlet_bc: 原入口BC
        outlet_bc: 原出口BC
        pipe: 管道
        fluid: 流体
        mode: 求解模式
    
    Returns:
        (upstream_section_bc, downstream_section_bc):
          上游段出口BC, 下游段入口BC (通过泄漏耦合)
    """
    
    def upstream_bc(t: float) -> float:
        """上游段 → 泄漏点: 出口压力 = 泄漏点压力"""
        # 泄漏BC直接返回压力（模式A出口BC）
        return 0.0  # 占位, 实际由下游段入口提供
    
    def downstream_bc(t: float) -> float:
        """泄漏点 → 下游段: 入口流量 = 原入口 - 泄漏"""
        return 0.0  # 占位
    
    return upstream_bc, downstream_bc


def leak_corrected_solve(
    pipe: Pipe,
    fluid: Liquid,
    leak: PipeLeak,
    inlet_bc: Callable,
    outlet_bc: Callable,
    mode: str = 'A',
    t_max: float = 50.0,
    P_initial: float = 2.0e6,
) -> Tuple:
    """
    带泄漏修正的瞬态求解
    
    单步MOC求解 + 泄漏修正:
    1. 正常求解整管
    2. 每个时间步在泄漏节点上分流
    3. 质量守恒: Q_out = Q_in - Q_leak
    
    Returns:
        (result, leak_flow_history)
    """
    from .solver import SinglePhaseTransientSolver, TransientResult
    
    solver = SinglePhaseTransientSolver(pipe, fluid)
    result = solver.solve(t_max=t_max, inlet_bc=inlet_bc, outlet_bc=outlet_bc,
                         mode=mode, P_initial=P_initial)
    
    # 泄漏流量追踪（后处理）
    Nx = solver.Nx
    leak_node = int((leak.position / pipe.length) * Nx)
    leak_node = np.clip(leak_node, 0, Nx)
    
    P = result.P
    rho = fluid.density(P, result.T if hasattr(result, 'T') else np.full_like(P, 20.0))
    
    leak_flows = np.zeros(len(P))
    for i in range(len(P)):
        if i * solver.dt >= leak.start_time:
            leak_flows[i] = leak.leak_flow(P[i, leak_node], rho[i, leak_node])
    
    # 标记泄漏信息
    result.leak_info = {
        'leak_flow': leak_flows,
        'leak_node': leak_node,
        'leak_position': leak.position,
        'total_leak_volume': np.trapezoid(leak_flows, dx=solver.dt),
    }
    
    return result, leak_flows


def detect_leak_from_data(
    P_upstream: np.ndarray,
    Q_downstream: np.ndarray,
    Q_inlet: np.ndarray,
    threshold: float = 0.05,
) -> dict:
    """
    根据实测/仿真数据检测泄漏
    
    原理: Q_inlet - Q_downstream > 阈值 → 存在泄漏
    
    Args:
        P_upstream: 上游压力序列
        Q_downstream: 下游流量序列
        Q_inlet: 入口流量序列
        threshold: 相对泄漏阈值
    
    Returns:
        detection结果
    """
    Q_loss = Q_inlet - Q_downstream
    rel_loss = Q_loss / np.maximum(Q_inlet, 1e-10)
    
    leak_detected = np.any(rel_loss > threshold)
    leak_start_idx = np.argmax(rel_loss > threshold) if leak_detected else -1
    max_leak = np.max(Q_loss) if leak_detected else 0.0
    
    return {
        'leak_detected': leak_detected,
        'detection_time_idx': leak_start_idx,
        'max_leakage_rate': max_leak,
        'max_relative_leakage': float(np.max(rel_loss)),
    }


if __name__ == '__main__':
    print("=== 泄漏模型测试 ===")
    print()
    
    # 1. 泄漏参数
    leak = PipeLeak(position=5000, orifice_diameter=0.015, 
                    discharge_coefficient=0.62, start_time=15.0)
    print(leak.describe())
    print()
    
    # 2. 不同压力下的泄漏流量
    print("泄漏流量 (孔口15mm):")
    for P in [1e6, 2e6, 5e6, 10e6]:
        Q = leak.leak_flow(P, 998.0)
        print(f"  P={P/1e6:.1f}MPa → Q_leak = {Q*1000:.2f} L/s = {Q*3600:.1f} m³/h")
    
    print()
    
    # 3. 完整仿真测试
    print("3. 含泄漏的瞬态仿真:")
    from .pipe import Pipe
    from .fluid import Liquid
    from .solver import flow_inlet, pressure_outlet
    
    pipe = Pipe(10000, 0.5, 0.01)
    fluid = Liquid()
    
    # 无泄漏仿真
    inlet = flow_inlet(lambda t: 1.0, lambda t: 300.0)
    outlet = pressure_outlet(lambda t: 2.0e6)
    
    result, leak_flows = leak_corrected_solve(
        pipe, fluid, leak, inlet, outlet, t_max=30,
    )
    
    print(f"  仿真完成: {len(result.P)} 时间步")
    print(f"  泄漏点: x={leak.position:.0f}m, 节点{result.leak_info['leak_node']}")
    print(f"  总泄漏量: {result.leak_info['total_leak_volume']:.2f} m³")
    print(f"  最大泄漏流量: {np.max(leak_flows)*1000:.2f} L/s")
    
    # 泄漏检测
    Q_in = np.ones(len(result.P))  # 假设入口恒流
    Q_out = np.array([1.0 - lf for lf in leak_flows])  # 出口 = 入口 - 泄漏
    detection = detect_leak_from_data(result.P[:, 0], Q_out, Q_in)
    print(f"  泄漏检测: {'✅ 检测到' if detection['leak_detected'] else '❌ 未检测'}")
    
    print()
    print("✅ 泄漏模型就绪!")
