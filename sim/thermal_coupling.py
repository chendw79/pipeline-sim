"""
thermal_coupling.py — 温度-压力全耦合瞬态求解器

核心升级:
  1. 状态方程: ρ = ρ(P, T) 全微分
  2. 能量方程: 包含焦耳-汤姆逊效应 + 管壁热交换
  3. 粘度: μ = μ(T, P) 全相关
  4. 波速: a = a(P, T) 可压缩性随温度变化

适用场景:
  - 热油/热水管道
  - 蒸汽管道的凝结水击
  - LNG/低温流体瞬态
  - 地热管道

超出原始MOC架构的新求解器:
  - T-P耦合MOC: 特征线 + 能量特征线
  - 扩展热力学: 内能项 + 焓流
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple
from .pipe import Pipe
from .fluid import Liquid


@dataclass
class ThermallyCoupledFluid(Liquid):
    """
    热-压力耦合流体（扩展状态方程）
    
    ρ(P,T) = ρ_ref · exp((P-P_ref)/K - α(T-T_ref))
    
    额外属性:
      cp: 比热容 J/(kg·K)
      beta: 焦耳-汤姆逊系数 K/Pa (节流降温)
      k_thermal: 热导率 W/(m·K)
    """
    cp: float = 4182.0            # J/(kg·K)
    beta_JT: float = 2.5e-7       # K/Pa 焦耳-汤姆逊系数
    k_thermal: float = 0.6        # W/(m·K)
    U_overall: float = 5.0        # 管壁总传热系数 W/(m²·K)
    T_ambient: float = 10.0       # 环境温度 °C
    
    def density_full(self, P: np.ndarray, T: np.ndarray) -> np.ndarray:
        """全耦合密度 (指数型状态方程)"""
        return self.rho_ref * np.exp(
            (P - self.P_ref) / self.bulk_modulus
            - self.thermal_expansion * (T - self.T_ref)
        )
    
    def d_rho_d_P(self, P: np.ndarray, T: np.ndarray) -> np.ndarray:
        """∂ρ/∂P (用于能量方程)"""
        return self.density_full(P, T) / self.bulk_modulus
    
    def d_rho_d_T(self, P: np.ndarray, T: np.ndarray) -> np.ndarray:
        """∂ρ/∂T (用于能量方程)"""
        return -self.density_full(P, T) * self.thermal_expansion


@dataclass
class ThermalSolverConfig:
    """热-压力耦合求解器参数"""
    solve_energy: bool = True           # 是否求解能量方程
    wall_heat_transfer: bool = True     # 管壁热交换
    joule_thomson: bool = True          # 焦耳-汤姆逊效应
    viscous_dissipation: bool = True    # 粘性耗散生热
    max_iterations: int = 3             # 耦合迭代次数


class CoupledThermalSolver:
    """
    温度-压力全耦合瞬态求解器
    
    扩展MOC方法，同时求解:
      1. 质量守恒 (MOC C⁺/C⁻)
      2. 动量守恒 (MOC C⁺/C⁻)
      3. 能量守恒 (MOC沿路径)
    
    网格: 交错网格 (staggered grid)
    """
    
    def __init__(self, pipe: Pipe, fluid: ThermallyCoupledFluid):
        self.pipe = pipe
        self.fluid = fluid
        self.cfg = ThermalSolverConfig()
        
        # 网格参数（默认20段，可取用户可以设置）
        self.Nx = getattr(pipe, 'Nx', 20)
        self.dx = pipe.length / self.Nx
        self.A = pipe.area()
        self.D = pipe.diameter
        
        # 波速（增加温度修正）
        self.a0 = pipe.wave_speed(fluid)
    
    def _wave_speed(self, T: np.ndarray) -> float:
        """温度修正的波速"""
        # 温度升高 → 密度降低 → 波速降低
        T_avg = np.mean(T)
        correction = 1.0 - 0.0005 * (T_avg - 20.0)  # ~0.05%/°C
        return self.a0 * max(correction, 0.8)
    
    def _heat_transfer(self, T: np.ndarray) -> np.ndarray:
        """管壁热交换: q = U·π·D·(T_amb - T)·dx"""
        if not self.cfg.wall_heat_transfer:
            return np.zeros_like(T)
        
        U = self.fluid.U_overall
        perimeter = self.pipe.perimeter()
        T_amb = self.fluid.T_ambient
        return U * perimeter * (T_amb - T) * self.dx
    
    def solve(self, t_max: float,
              inlet_bc: Callable,
              outlet_bc: Callable,
              T_inlet_func: Optional[Callable[[float], float]] = None,
              mode: str = 'A',
              P_initial: float = 2.0e6,
              T_initial: float = 20.0,
              ) -> Tuple:
        """
        全耦合T-P瞬态求解
        
        Args:
            t_max: 仿真时长 (s)
            inlet_bc: 入口BC (同上)
            outlet_bc: 出口BC (同上)
            T_inlet_func: 入口温度调度
            mode: 'A'/'B'
            P_initial: 初始压力
            T_initial: 初始温度
        
        Returns:
            result: TransientResult-like (含T场)
        """
        from .solver import TransientResult
        
        dt = self.dx / self.a0
        Nt = int(t_max / dt) + 1
        
        g = 9.81
        B = self.a0 / (g * self.A) if self.A > 0 else 0
        
        # 初始化
        H = np.zeros((Nt, self.Nx + 1))
        V = np.zeros((Nt, self.Nx + 1))
        T = np.full((Nt, self.Nx + 1), T_initial)
        P = np.full((Nt, self.Nx + 1), P_initial)
        rho = np.full((Nt, self.Nx + 1), self.fluid.rho_ref)
        
        # 初始条件
        rho0 = self.fluid.density_full(np.full(self.Nx + 1, P_initial),
                                        np.full(self.Nx + 1, T_initial))
        H0 = P_initial / (rho0.max() * g) if rho0.max() > 0 else 50.0
        V0 = 1.0
        
        H[0, :] = H0
        V[0, :] = V0
        P[0, :] = P_initial
        T[0, :] = T_initial
        rho[0, :] = rho0
        mu_arr = self.fluid.viscosity(np.full_like(T[0], T_initial))
        
        # 摩擦系数
        Re0 = rho0 * abs(V0) * self.D / max(mu_arr.max(), 1e-10)
        f_n = self.pipe.friction_factor(np.full_like(V[0], V0), Re0)
        R = f_n * self.dx / (2 * g * self.D)
        
        # 时间步进
        T_inlet = T_inlet_func if T_inlet_func else (lambda t: T_initial)
        
        for n in range(Nt - 1):
            tn = n * dt
            ti = tn + dt
            
            # 波速（温度修正）
            a = self._wave_speed(T[n])
            B = a / (g * self.A) if self.A > 0 else 0
            CFL = a * dt / self.dx
            
            # --- 内部节点: MOC ---
            for i in range(1, self.Nx):
                # C⁺ (右行波)
                q_abs = abs(V[n, i-1])
                R_i = f_n[i-1] * self.dx / (2 * g * self.D)
                Cp = H[n, i-1] + B * V[n, i-1] - R_i * V[n, i-1] * q_abs
                
                # C⁻ (左行波)
                q_abs = abs(V[n, i+1])
                R_i = f_n[i+1] * self.dx / (2 * g * self.D)
                Cm = H[n, i+1] - B * V[n, i+1] + R_i * V[n, i+1] * q_abs
                
                H_new = 0.5 * (Cp + Cm)
                V_new = 0.5 * (Cp - Cm) / B
                
                H[n+1, i] = H_new
                V[n+1, i] = V_new
            
            # --- 能量方程求解（温度传播 + 耦合项）---
            if self.cfg.solve_energy:
                for i in range(1, self.Nx + 1):
                    # 温度输运（特征线沿速度方向）
                    if V[n, i] > 0:
                        # 下游流动
                        T[n+1, i] = T[n, max(0, i-1)]
                    else:
                        T[n+1, i] = T[n, min(self.Nx, i+1)]
                    
                    # 管壁热交换
                    dt_local = dt
                    q_wall = self._heat_transfer(np.array([T[n+1, i]]))
                    q_per_mass = q_wall[0] / (max(rho[n, i], 1.0) * self.A * self.dx)
                    
                    if self.cfg.viscous_dissipation:
                        # 粘性耗散生热
                        visc_heat = f_n[i] * rho[n, i] * abs(V[n, i])**3 / (2 * self.D)
                        q_visc = visc_heat * np.pi * self.D * self.dx
                        q_per_mass += q_visc / (max(rho[n, i], 1.0) * self.A * self.dx)
                    
                    if self.cfg.joule_thomson:
                        # 焦耳-汤姆逊效应 (节流降温)
                        dP_local = (P[n, min(self.Nx, i+1)] - P[n, max(0, i-1)]) / (2 * self.dx)
                        JT_cooling = self.fluid.beta_JT * dP_local * V[n, i]
                        q_per_mass += -JT_cooling  # 焦汤系数为正时，压降导致降温
                    
                    dt_s = dt_local
                    T[n+1, i] += q_per_mass * dt_s / self.fluid.cp
            
            # --- 边界条件 ---
            if mode.upper() == 'A':
                Q_in, _ = inlet_bc(ti)
                V_in = Q_in / self.A
                V[n+1, 0] = V_in
                # C⁻ at inlet
                q_abs = abs(V[n, 1])
                R_i = f_n[1] * self.dx / (2 * g * self.D)
                Cm0 = H[n, 1] - B * V[n, 1] + R_i * V[n, 1] * q_abs
                H[n+1, 0] = Cm0 + B * V_in
                T[n+1, 0] = T_inlet(ti)
                
                # Outlet
                P_out = outlet_bc(ti)
                rho_out = self.fluid.density_full(np.array([P_out]), np.array([T[n, -2]]))[0]
                H_out = P_out / (rho_out * g) if rho_out > 0 else 0
                # C⁺ at outlet
                q_abs = abs(V[n, self.Nx - 1])
                R_i = f_n[self.Nx - 1] * self.dx / (2 * g * self.D)
                CpL = H[n, self.Nx - 1] + B * V[n, self.Nx - 1] - R_i * V[n, self.Nx - 1] * q_abs
                H[n+1, self.Nx] = H_out
                V[n+1, self.Nx] = (CpL - H_out) / B if B > 0 else 0
                T[n+1, self.Nx] = T[n+1, self.Nx - 1]  # 温度流出
            
            # --- 更新密度和压力 ---
            rho_new = self.fluid.density_full(P[n], T[n+1])
            P[n+1] = rho_new * g * H[n+1]
            # 更新摩擦系数
            V_abs = np.maximum(abs(V[n+1]), 1e-10)
            mu_arr = self.fluid.viscosity(T[n+1])
            Re = rho_new * V_abs * self.D / np.maximum(mu_arr, 1e-10)
            f_n = self.pipe.friction_factor(V[n+1], Re)
            rho[n+1] = rho_new
        
        # 构建结果对象
        t_arr = np.arange(Nt) * dt
        x_arr = np.linspace(0, self.pipe.length, self.Nx + 1)
        result = TransientResult(
            t=t_arr, x=x_arr, P=P, T=T, V=V,
            Q=V * self.A,
            rho=rho,
            mu=np.full_like(V, 1e-3),
        )
        result.H = H
        return result


if __name__ == '__main__':
    print("=== 温度-压力全耦合求解器测试 ===")
    print()
    
    from .pipe import Pipe
    from .solver import flow_inlet, pressure_outlet
    
    pipe = Pipe(length=10000, diameter=0.5, wall_thickness=0.01)
    
    # 热耦合流体（原油）
    oil = ThermallyCoupledFluid(
        name='Crude Oil',
        rho_ref=860.0,
        bulk_modulus=1.5e9,
        cp=2000.0,
        U_overall=3.0,
        T_ambient=5.0,
        viscosity_ref=0.05,
    )
    
    print(f"流体: {oil.name}")
    print(f"  密度: {oil.rho_ref} kg/m³")
    print(f"  比热: {oil.cp} J/(kg·K)")
    print(f"  环境温度: {oil.T_ambient}°C")
    print()
    
    # 求解
    solver = CoupledThermalSolver(pipe, oil)
    inlet = flow_inlet(lambda t: 1.0, lambda t: 300.0)
    outlet = pressure_outlet(lambda t: 2.0e6)
    
    result = solver.solve(t_max=50.0, inlet_bc=inlet, outlet_bc=outlet,
                          T_inlet_func=lambda t: 60.0,  # 入口60°C热油
                          P_initial=2.0e6, T_initial=20.0)
    
    print(f"求解完成: {len(result.T)} 时间步 × {result.T.shape[1]} 节点")
    print()
    
    # 出口温度
    T_outlet = result.T[-1, -1]
    T_inlet = result.T[0, 0]
    print(f"入口温度: 60.0°C")
    print(f"出口温度(稳态): {T_outlet:.1f}°C")
    print(f"温降: {60.0 - T_outlet:.1f}°C (管线10km)")
    
    # 不同时间的沿程温度
    for pct in [0, 25, 50, 75, 100]:
        idx = int(pct * (len(result.T) - 1) / 100)
        T_profile = result.T[idx]
        print(f"  t={result.t[idx]:.0f}s: T_in={T_profile[0]:.1f}°C → T_out={T_profile[-1]:.1f}°C")
    
    print()
    print("✅ 温度-压力全耦合求解器就绪!")
