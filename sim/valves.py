"""
valves.py — 阀门特性曲线 + PID控制器 + 调节阀模型

包含:
  1. ValveCv — 阀门流量系数模型 (Cv/Kv)
  2. PIDController — 比例-积分-微分控制器
  3. ControlValve — 调节阀（含执行器动态）
  4. 边界条件工厂函数: valve_bc, pid_control_bc

标准阀门方程:
  Q = Cv · f(pos) · sqrt(ΔP / SG)
  其中 Cv: 全开流量系数, pos: [0,1] 阀位, f(pos): 流量特性
"""

import numpy as np
from typing import Callable, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ValveCv:
    """
    阀门流量系数模型
    
    支持的流量特性:
      - 'linear':      f(pos) = pos
      - 'equal_pct':   f(pos) = R^(pos-1), R=可调比
      - 'quick_open':  f(pos) = sqrt(pos)
      - 'parabolic':   f(pos) = pos²
    """
    Cv: float = 50.0          # 全开流量系数 (US gal/min at 1psi ΔP)
    Kv: Optional[float] = None  # 全开流量系数 (m³/h at 1bar ΔP)
    characteristic: str = 'equal_pct'
    rangeability: float = 50.0  # 可调比 (仅 equal_pct)
    
    def __post_init__(self):
        if self.Kv is not None:
            self.Cv = self.Kv * 1.156  # Kv → Cv 转换
    
    def flow_coefficient(self, position: float) -> float:
        """计算当前开度下的有效Cv"""
        pos = np.clip(position, 0.0, 1.0)
        
        if self.characteristic == 'linear':
            f = pos
        elif self.characteristic == 'equal_pct':
            f = self.rangeability ** (pos - 1) if pos > 0 else 0.0
        elif self.characteristic == 'quick_open':
            f = np.sqrt(pos) if pos > 0 else 0.0
        elif self.characteristic == 'parabolic':
            f = pos ** 2
        else:
            f = pos  # fallback to linear
        
        return float(self.Cv * f)
    
    def flow_rate(self, position: float, delta_P: float, SG: float = 1.0) -> float:
        """
        计算通过阀门的流量 (m³/s)
        
        Args:
            position: 阀位 [0, 1]
            delta_P: 压降 (Pa)
            SG: 比重 (相对于水, water=1.0)
        
        Returns:
            Q: 流量 (m³/s)
        """
        # Cv单位: US gal/min at 1 psi
        # Q(gpm) = Cv · sqrt(ΔP(psi) / SG)
        # ΔP从Pa转换到psi
        dP_psi = delta_P / 6894.76
        dP_psi = max(dP_psi, 0.0)  # 防止负压降
        
        Cv_eff = self.flow_coefficient(position)
        Q_gpm = Cv_eff * np.sqrt(max(dP_psi / SG, 0.0))
        
        # 转换为 m³/s
        return float(Q_gpm * 6.309e-5)
    
    def pressure_drop(self, position: float, Q: float, SG: float = 1.0) -> float:
        """计算给定流量下的压降 (Pa)"""
        Cv_eff = self.flow_coefficient(position)
        if Cv_eff <= 0:
            return 1e10  # 完全关闭
        
        Q_gpm = Q / 6.309e-5
        dP_psi = SG * (Q_gpm / Cv_eff) ** 2
        return float(dP_psi * 6894.76)


@dataclass
class PIDController:
    """
    数字PID控制器 (位置型)
    
    u(t) = Kp · e(t) + Ki · ∫e dt + Kd · de/dt
    
    支持:
      - 积分饱和 (anti-windup)
      - 输出限幅
      - 微分滤波 (减少噪声放大)
    """
    Kp: float = 1.0        # 比例增益
    Ki: float = 0.1        # 积分增益
    Kd: float = 0.05       # 微分增益
    setpoint: float = 0.0  # 设定值
    output_min: float = 0.0   # 输出下限
    output_max: float = 1.0   # 输出上限
    dt: float = 0.1        # 控制周期 (s)
    tau_d: float = 0.1     # 微分滤波时间常数
    
    # 内部状态
    integral: float = 0.0
    prev_error: float = 0.0
    prev_derivative: float = 0.0
    
    def reset(self):
        """重置PID状态"""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0
    
    def update(self, measurement: float) -> float:
        """
        单步更新
        
        Args:
            measurement: 当前测量值
        
        Returns:
            u: 控制输出
        """
        error = self.setpoint - measurement
        
        # P项
        P = self.Kp * error
        
        # I项 (含anti-windup)
        self.integral += error * self.dt
        
        # 条件积分 (conditional integration): 在饱和时停止积分
        u_prev = P + self.Ki * self.integral + self.Kd * self.prev_derivative
        if not (u_prev <= self.output_min and error < 0) and \
           not (u_prev >= self.output_max and error > 0):
            pass  # 正常积分
        else:
            self.integral -= error * self.dt  # 撤销积分
        
        I = self.Ki * self.integral
        
        # D项 (含滤波)
        derivative = (error - self.prev_error) / self.dt
        D_filtered = (self.dt * derivative + self.tau_d * self.prev_derivative) / \
                     (self.dt + self.tau_d) if self.dt + self.tau_d > 0 else 0.0
        D = self.Kd * D_filtered
        
        # 输出
        u = P + I + D
        
        # 限幅
        u = np.clip(u, self.output_min, self.output_max)
        
        # 更新状态
        self.prev_error = error
        self.prev_derivative = D_filtered
        
        return float(u)


@dataclass
class ControlValve:
    """
    调节阀 (含执行器动态)
    
    模型:
      - 一阶滞后: d(pos)/dt = (pos_set - pos) / τ_act
      - 行程限位
      - 粘滞/死区
    """
    valve: ValveCv = field(default_factory=ValveCv)
    tau_actuator: float = 2.0      # 执行器时间常数 (s)
    travel_time: float = 10.0      # 全行程时间 (s) (override tau if >0)
    deadband: float = 0.005        # 死区 (开度百分比)
    position: float = 0.5          # 当前开度 [0,1]
    target_position: float = 0.5   # 目标开度 [0,1]
    position_min: float = 0.0
    position_max: float = 1.0
    
    # 内部状态
    stuck: bool = False
    stick_slip: float = 0.02       # 粘滞幅度
    
    def set_position(self, pos: float):
        """设置目标开度"""
        self.target_position = np.clip(pos, self.position_min, self.position_max)
    
    def step(self, dt: float) -> float:
        """
        执行器动态更新
        
        Returns:
            position: 更新后的阀门开度 [0, 1]
        """
        # 死区检查
        if abs(self.target_position - self.position) < self.deadband:
            return float(self.position)
        
        # 粘滞效应
        if abs(self.target_position - self.position) < self.stick_slip:
            # 静摩擦，不动作
            return float(self.position)
        
        # 一阶滞后动态
        tau = self.travel_time if self.travel_time > 0 else self.tau_actuator
        if tau > 0:
            # 限制最大移动速度
            max_move = dt / tau  # 每个时间步的最大开度变化
            delta = self.target_position - self.position
            delta = np.clip(delta, -max_move, max_move)
            self.position += delta
        
        self.position = np.clip(self.position, self.position_min, self.position_max)
        return float(self.position)


# ── 边界条件工厂函数 ──────────────────────────────────

def valve_bc(
    valve: ControlValve,
    upstream_pressure_func: Callable[[float], float],
    downstream_pressure_func: Callable[[float], float],
    SG: float = 1.0,
    position_schedule: Optional[Callable[[float], float]] = None,
) -> Callable[[float], Tuple[float, float]]:
    """
    阀门边界条件工厂
    
    根据阀门开度 + 上下游压力 → 计算通过阀门的流量和出口压力
    适用于管段之间的连接。
    
    Args:
        valve: ControlValve对象
        upstream_pressure_func: 上游压力函数 P_up(t)
        downstream_pressure_func: 下游压力函数 P_down(t)
        SG: 比重
        position_schedule: 阀门开度调度函数 pos(t), 默认使用valve.target_position
    
    Returns:
        bc(t) → (Q_through, P_downstream)
    """
    def bc(t: float) -> Tuple[float, float]:
        # 阀门开度
        pos = position_schedule(t) if position_schedule else valve.position
        
        # 上下游压力
        P_up = upstream_pressure_func(t)
        P_down = downstream_pressure_func(t)
        
        # 压降
        delta_P = max(P_up - P_down, 0.0)
        
        # 流量
        Q = valve.valve.flow_rate(pos, delta_P, SG)
        
        # 更新执行器位置
        valve.step(valve.valve.Cv * dt if hasattr(valve, '_dt') else 0.1)
        
        return Q, P_down
    
    return bc


def pid_control_bc(
    pid: PIDController,
    valve: ControlValve,
    measurement_func: Callable[[float], float],
    mode: str = 'A',
    outlet_pressure_func: Optional[Callable[[float], float]] = None,
    SG: float = 1.0,
) -> Callable[[float], Tuple[float, float]]:
    """
    PID控制阀边界条件
    
    根据设定值和测量值 → PID计算阀位 → 阀门流量
    
    Args:
        pid: PID控制器
        valve: 调节阀
        measurement_func: 测量值函数 m(t) (例如下游压力)
        mode: 控制模式
        outlet_pressure_func: 出口压力函数 (Mode A)
        SG: 比重
    
    Returns:
        bc(t) → (Q(t), T(t))
    """
    def bc(t: float) -> Tuple[float, float]:
        # 读取测量值
        meas = measurement_func(t)
        
        # PID计算阀位
        pos = pid.update(meas)
        valve.set_position(pos)
        
        # 执行器动态
        current_pos = valve.step(pid.dt)
        
        if mode.upper() == 'A':
            # 需要出口压力来计算流量
            P_out = outlet_pressure_func(t) if outlet_pressure_func else 2.0e6
            
            # 简化：假设上游压力 = 泵出口定压，下游 = 管线末端
            Q = valve.valve.flow_rate(current_pos, 1.0e5, SG)  # 示例压降0.1MPa
            
            return Q, 300.0  # Q, T
        else:
            return P_out, 300.0
    
    return bc


def step_valve_schedule(
    initial_pos: float, 
    final_pos: float, 
    t_start: float, 
    duration: float = 0.0
) -> Callable[[float], float]:
    """
    生成阀门开度阶跃/斜坡调度函数
    
    Args:
        initial_pos: 初始开度 [0,1]
        final_pos:  最终开度 [0,1]
        t_start:    开始时间 (s)
        duration:   过渡时间 (s), 0=阶跃
    
    Returns:
        pos(t): 阀门开度随时间变化
    """
    if duration <= 0:
        return lambda t: final_pos if t >= t_start else initial_pos
    else:
        def ramp(t: float) -> float:
            if t <= t_start:
                return initial_pos
            elif t >= t_start + duration:
                return final_pos
            else:
                frac = (t - t_start) / duration
                return initial_pos + (final_pos - initial_pos) * frac
        return ramp


if __name__ == '__main__':
    print("=== 阀门 & PID 测试 ===")
    print()
    
    # 1. ValveCv
    print("1. 阀门流量系数 (Cv=50, equal_pct, R=50)")
    v = ValveCv(Cv=50.0, characteristic='equal_pct', rangeability=50.0)
    for pos in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
        Cv_eff = v.flow_coefficient(pos)
        print(f"  开度 {pos*100:5.0f}% → Cv_eff = {Cv_eff:.3f}")
    
    print()
    print("2. 压降0.1MPa时的流量:")
    dP = 100000  # 0.1 MPa
    for pos in [0.25, 0.5, 0.75, 1.0]:
        Q = v.flow_rate(pos, dP)
        print(f"  开度 {pos*100:5.0f}% → Q = {Q:.4f} m³/s = {Q*3600:.1f} m³/h")
    
    # 2. PID
    print()
    print("3. PID控制器 (设定值=2.0MPa, Kp=0.5, Ki=0.05)")
    pid = PIDController(Kp=0.5, Ki=0.05, Kd=0.01, setpoint=2.0e6,
                        output_min=0.0, output_max=1.0, dt=0.1)
    valve = ControlValve()
    
    # 模拟: 从1.5MPa → 设定值2.0MPa
    P = 1.5e6
    print(f"  初始压力: {P/1e6:.2f} MPa")
    for step_i in range(20):
        pos = pid.update(P)
        valve.set_position(pos)
        v_pos = valve.step(0.1)
        # 简化响应: 阀门开大 → 流量增大 → 压力上升
        P += (pos - 0.3) * 50000
        P = min(P, 2.1e6)
        if step_i % 5 == 0:
            print(f"  t={step_i*0.1:.1f}s P={P/1e6:.3f}MPa pos={pos:.3f}")
    
    print()
    print("4. 阀门调度 (慢关: 5~8秒从1→0)")
    schedule = step_valve_schedule(1.0, 0.0, 5.0, 3.0)
    for t in [0, 3, 5, 6, 7, 8, 10]:
        print(f"  t={t}s → pos={schedule(t):.3f}")
    
    print()
    print("✅ 阀门 & PID 模型就绪!")
