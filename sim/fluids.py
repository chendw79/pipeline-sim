"""
fluids.py — Non-Newtonian fluid models for pipeline simulation

支持的模型:
  1. PowerLawFluid — 幂律流体 (τ = K·γ̇ⁿ)
  2. BinghamFluid — 宾汉流体 (τ = τ₀ + μ·γ̇)
  3. HerschelBulkleyFluid — 赫巴流体 (τ = τ₀ + K·γ̇ⁿ)
  
所有模型继承自Liquid基类，重写viscosity为表观粘度函数。
摩擦系数计算使用对应的雷诺数（Metzner-Reed雷诺数）。
"""

import numpy as np
from dataclasses import dataclass
from .fluid import Liquid


def _apparent_viscosity_powerlaw(K: float, n: float, V: np.ndarray, D: float) -> np.ndarray:
    """幂律流体表观粘度 (Metzner-Reed)"""
    # γ̇_w = (3n+1)/(4n) * 8V/D (壁面剪切率)
    gamma_w = (3 * n + 1) / (4 * n) * 8 * np.abs(V) / D
    # 表观粘度 μ_app = K · γ̇^(n-1)
    return K * np.maximum(gamma_w, 1e-10) ** (n - 1)


def _apparent_viscosity_bingham(tau0: float, mu_p: float, V: np.ndarray, D: float) -> np.ndarray:
    """宾汉流体表观粘度"""
    # Hedström模型
    gamma_w = 8 * np.abs(V) / D
    # μ_app = μ_p + τ₀ / γ̇   (当γ̇足够大)
    # 当 τ₀ > μ_p·γ̇ 时，流体不流动（屈服）
    tau_w = tau0 + mu_p * gamma_w
    return np.where(gamma_w > 1e-10, tau_w / gamma_w, 1e10)


def _apparent_viscosity_hb(tau0: float, K: float, n: float, V: np.ndarray, D: float) -> np.ndarray:
    """赫巴流体表观粘度"""
    gamma_w = (3 * n + 1) / (4 * n) * 8 * np.abs(V) / D
    tau_w = tau0 + K * np.maximum(gamma_w, 1e-10) ** n
    return np.where(gamma_w > 1e-10, tau_w / gamma_w, 1e10)


@dataclass
class PowerLawFluid(Liquid):
    """
    幂律流体 (Ostwald-de Waele)
    τ = K · γ̇ⁿ
    
    n = 1: 牛顿流体 (K = μ)
    n < 1: 剪切稀化（假塑性）
    n > 1: 剪切增稠（胀流性）
    """
    consistency_index: float = 0.1      # K, Pa·sⁿ
    flow_behavior_index: float = 0.8    # n, 无量纲
    # 参考密度（不受剪切影响）
    rho_ref: float = 900.0
    
    def __post_init__(self):
        self.name = f"PowerLaw(n={self.flow_behavior_index:.2f})"
    
    def viscosity(self, T: np.ndarray) -> np.ndarray:
        """表观粘度 - 需要速度梯度信息，这里仅做温度修正"""
        # PowerLaw粘度主要由剪切率决定，温度修正作为次要项
        T_clipped = np.clip(T, -50.0, 200.0)
        temp_factor = np.exp(-self.visc_T_coeff * (T_clipped - self.T_ref))
        return self.consistency_index * np.ones_like(T) * temp_factor
    
    def apparent_viscosity(self, V: np.ndarray, D: float, T: np.ndarray) -> np.ndarray:
        """计算表观粘度（含剪切率效应）"""
        mu_shear = _apparent_viscosity_powerlaw(
            self.consistency_index, self.flow_behavior_index, V, D)
        T_clipped = np.clip(T, -50.0, 200.0)
        temp_factor = np.exp(-self.visc_T_coeff * (T_clipped - self.T_ref))
        return mu_shear * temp_factor
    
    def friction_factor(self, V: np.ndarray, Re: np.ndarray, D: float) -> np.ndarray:
        """幂律流体摩擦系数 (Dodge-Metzner 关联式)"""
        # Metzner-Reed 雷诺数
        V_abs = np.maximum(np.abs(V), 1e-10)
        mu_app = _apparent_viscosity_powerlaw(
            self.consistency_index, self.flow_behavior_index, V, D)
        Re_MR = self.rho_ref * V_abs * D / np.maximum(mu_app, 1e-10)
        
        Re_MR = np.maximum(Re_MR, 1.0)
        n = self.flow_behavior_index
        
        # Dodge-Metzner: 1/√f = (4/n^0.75) · log(Re_MR · f^(1-n/2)) - 0.4/n^1.2
        # 迭代近似（Blasius型）
        f = np.where(Re_MR < 2100,
                     64.0 / Re_MR,  # 层流: f = 64/Re_MR
                     0.079 * Re_MR ** (-0.25) * (3 * n + 1) / (4 * n) ** 0.25  # 湍流近似
                     )
        return f


@dataclass
class BinghamFluid(Liquid):
    """
    宾汉流体
    τ = τ₀ + μₚ · γ̇
    
    用于: 泥浆、牙膏、沥青、部分原油
    """
    yield_stress: float = 5.0          # τ₀, Pa (屈服应力)
    plastic_viscosity: float = 0.05    # μₚ, Pa·s (塑性粘度)
    rho_ref: float = 1050.0
    
    def __post_init__(self):
        self.name = f"Bingham(tau0={self.yield_stress:.1f})"
    
    def viscosity(self, T: np.ndarray) -> np.ndarray:
        """基础粘度（温度修正后的塑性粘度）"""
        T_clipped = np.clip(T, -50.0, 200.0)
        temp_factor = np.exp(-self.visc_T_coeff * (T_clipped - self.T_ref))
        return self.plastic_viscosity * np.ones_like(T) * temp_factor
    
    def apparent_viscosity(self, V: np.ndarray, D: float, T: np.ndarray) -> np.ndarray:
        """表观粘度（含屈服应力效应）"""
        mu_app = _apparent_viscosity_bingham(
            self.yield_stress, self.plastic_viscosity, V, D)
        T_clipped = np.clip(T, -50.0, 200.0)
        temp_factor = np.exp(-self.visc_T_coeff * (T_clipped - self.T_ref))
        return mu_app * temp_factor
    
    def friction_factor(self, V: np.ndarray, Re: np.ndarray, D: float) -> np.ndarray:
        """宾汉流体摩擦系数 (Hedström模型)"""
        V_abs = np.maximum(np.abs(V), 1e-10)
        mu_app = _apparent_viscosity_bingham(
            self.yield_stress, self.plastic_viscosity, V, D)
        
        # Hedström数
        He = self.rho_ref * self.yield_stress * D**2 / np.maximum(self.plastic_viscosity**2, 1e-10)
        # 修正临界雷诺数 (Hedström)
        Re_crit = 2100 * (1 + He / 3000) ** 0.5
        
        Re_eff = self.rho_ref * V_abs * D / np.maximum(mu_app, 1e-10)
        Re_eff = np.maximum(Re_eff, 1.0)
        
        f = np.where(Re_eff < Re_crit,
                     64.0 / Re_eff,  # 层流
                     0.079 * Re_eff ** (-0.25)  # 湍流
                     )
        return f


@dataclass
class HerschelBulkleyFluid(Liquid):
    """
    赫巴流体 (Herschel-Bulkley)
    τ = τ₀ + K · γ̇ⁿ
    
    幂律 + 屈服应力的通用模型，能拟合大多数非牛顿流体
    """
    yield_stress: float = 5.0          # τ₀, Pa
    consistency_index: float = 0.1      # K
    flow_behavior_index: float = 0.8    # n
    rho_ref: float = 950.0
    
    def __post_init__(self):
        self.name = f"HerschelBulkley(tau0={self.yield_stress:.1f},n={self.flow_behavior_index:.2f})"
    
    def viscosity(self, T: np.ndarray) -> np.ndarray:
        T_clipped = np.clip(T, -50.0, 200.0)
        temp_factor = np.exp(-self.visc_T_coeff * (T_clipped - self.T_ref))
        return self.consistency_index * np.ones_like(T) * temp_factor
    
    def apparent_viscosity(self, V: np.ndarray, D: float, T: np.ndarray) -> np.ndarray:
        mu_app = _apparent_viscosity_hb(
            self.yield_stress, self.consistency_index,
            self.flow_behavior_index, V, D)
        T_clipped = np.clip(T, -50.0, 200.0)
        temp_factor = np.exp(-self.visc_T_coeff * (T_clipped - self.T_ref))
        return mu_app * temp_factor
    
    def friction_factor(self, V: np.ndarray, Re: np.ndarray, D: float) -> np.ndarray:
        """赫巴流体摩擦系数（简化的Metzner-Reed + Hedström组合）"""
        V_abs = np.maximum(np.abs(V), 1e-10)
        mu_app = _apparent_viscosity_hb(
            self.yield_stress, self.consistency_index,
            self.flow_behavior_index, V, D)
        
        Re_MR = self.rho_ref * V_abs * D / np.maximum(mu_app, 1e-10)
        Re_MR = np.maximum(Re_MR, 1.0)
        
        f = np.where(Re_MR < 2100,
                     64.0 / Re_MR,
                     0.079 * Re_MR ** (-0.25)
                     )
        return f


def describe_fluid(fluid: Liquid) -> str:
    """描述流体类型和关键参数"""
    lines = [f"流体类型: {fluid.name}"]
    lines.append(f"  密度: {fluid.rho_ref:.0f} kg/m³")
    
    if isinstance(fluid, PowerLawFluid):
        lines.append(f"  模型: 幂律 τ = K·γ̇ⁿ")
        lines.append(f"  K (稠度系数): {fluid.consistency_index:.3f} Pa·sⁿ")
        lines.append(f"  n (流变指数): {fluid.flow_behavior_index:.3f}")
        if fluid.flow_behavior_index < 1:
            lines.append(f"  特性: 剪切稀化 (假塑性)")
        elif fluid.flow_behavior_index > 1:
            lines.append(f"  特性: 剪切增稠 (胀流性)")
        else:
            lines.append(f"  特性: 牛顿流体")
    
    elif isinstance(fluid, BinghamFluid):
        lines.append(f"  模型: 宾汉 τ = τ₀ + μₚ·γ̇")
        lines.append(f"  τ₀ (屈服应力): {fluid.yield_stress:.1f} Pa")
        lines.append(f"  μₚ (塑性粘度): {fluid.plastic_viscosity:.5f} Pa·s")
    
    elif isinstance(fluid, HerschelBulkleyFluid):
        lines.append(f"  模型: 赫巴 τ = τ₀ + K·γ̇ⁿ")
        lines.append(f"  τ₀ (屈服应力): {fluid.yield_stress:.1f} Pa")
        lines.append(f"  K (稠度系数): {fluid.consistency_index:.4f}")
        lines.append(f"  n (流变指数): {fluid.flow_behavior_index:.3f}")
    
    return '\n'.join(lines)


if __name__ == '__main__':
    # 测试非牛顿流体
    power = PowerLawFluid(consistency_index=0.15, flow_behavior_index=0.75)
    bingham = BinghamFluid(yield_stress=8.0, plastic_viscosity=0.03)
    hb = HerschelBulkleyFluid(yield_stress=5.0, consistency_index=0.12, flow_behavior_index=0.8)
    
    print("=== 非牛顿流体测试 ===")
    print()
    
    for fluid in [power, bingham, hb]:
        print(describe_fluid(fluid))
        # 测试不同流速下的表观粘度
        V = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
        D = 0.5
        T = np.array([20.0] * 5)
        mu_app = fluid.apparent_viscosity(V, D, T)
        f = fluid.friction_factor(V, np.ones_like(V), D)
        for vi, mui, fi in zip(V, mu_app, f):
            print(f"  V={vi:.1f}m/s → μ_app={mui:.4f} Pa·s, f={fi:.4f}")
        print()
    
    print("✅ 非牛顿流体模型就绪!")
