"""
fluid.py — Liquid fluid properties with P,T dependence
"""
import numpy as np
from dataclasses import dataclass

@dataclass
class Liquid:
    """Liquid with pressure and temperature dependent properties"""
    name: str = "Water"
    # Reference state
    rho_ref: float = 998.0        # kg/m³ at P_ref, T_ref
    P_ref: float = 1.01325e5      # Pa (atmospheric)
    T_ref: float = 20.0           # °C
    # Compressibility
    bulk_modulus: float = 2.15e9  # Pa (isothermal bulk modulus)
    # Thermal
    thermal_expansion: float = 2.07e-4  # 1/K (volumetric expansion coeff)
    cp: float = 4182.0            # J/(kg·K) specific heat
    viscosity_ref: float = 1.0e-3 # Pa·s at T_ref
    # Viscosity model parameters (exponential)
    visc_T_coeff: float = 0.02    # 1/K (for μ = μ_ref * exp(-c*(T-T_ref)))
    
    def density(self, P: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Density as function of pressure and temperature"""
        # Linearized state equation
        dP = P - self.P_ref
        dT = T - self.T_ref
        return self.rho_ref * (1.0 + dP / self.bulk_modulus - self.thermal_expansion * dT)
    
    def viscosity(self, T: np.ndarray) -> np.ndarray:
        """Temperature-dependent viscosity (clipped to prevent overflow)"""
        # Clip temperature to prevent exp overflow
        T_clipped = np.clip(T, -50.0, 200.0)
        result = self.viscosity_ref * np.exp(-self.visc_T_coeff * (T_clipped - self.T_ref))
        # Clip viscosity to physically valid range
        return np.clip(result, 1e-6, 100.0)
    
    def head_to_pressure(self, H: np.ndarray, rho: np.ndarray) -> np.ndarray:
        """Convert head (m) to pressure (Pa)"""
        return rho * 9.81 * H
    
    def pressure_to_head(self, P: np.ndarray, rho: np.ndarray) -> np.ndarray:
        """Convert pressure (Pa) to head (m)"""
        return P / (rho * 9.81)
