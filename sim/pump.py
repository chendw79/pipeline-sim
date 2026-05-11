"""
Pump component model for PipelineSim.

Models centrifugal pumps (the most common type in pipeline applications).
Provides head-flow curves and power consumption calculations.
"""

import numpy as np
from typing import Tuple, Callable, Optional
from dataclasses import dataclass


@dataclass
class Pump:
    """
    Centrifugal pump model using affinity laws.
    
    Reference: ISO 9906 / API 610
    
    Head-flow characteristic (quadratic fit):
        H(Q) = a·Q² + b·Q + c
        
    Efficiency curve:
        η(Q) = α·Q² + β·Q + γ
        
    Affinity Laws (for variable speed):
        Q₂/Q₁ = N₂/N₁
        H₂/H₁ = (N₂/N₁)²
        P₂/P₁ = (N₂/N₁)³
    """
    
    name: str = "Centrifugal Pump"
    
    # Rated conditions
    Q_rated: float = 0.25        # m³/s (rated flow)
    H_rated: float = 500.0      # m (rated head)
    N_rated: float = 1500.0     # RPM (rated speed)
    efficiency_rated: float = 0.82  # at BEP
    
    # Head curve coefficients: H(Q) = a·Q² + b·Q + c (at rated speed)
    # Default: typical medium-size centrifugal pump
    h_a: float = -1500.0    # Q² coefficient
    h_b: float = -20.0      # Q coefficient  
    h_c: float = 550.0      # constant
    
    # Efficiency curve coefficients: η(Q) = α·Q² + β·Q + γ
    e_a: float = -6.0       # Q² coefficient
    e_b: float = 3.0        # Q coefficient
    e_c: float = 0.3        # constant
    
    # Fluid properties
    fluid_density: float = 860.0  # kg/m³
    
    def __post_init__(self):
        # Validate rated point is on the curve
        H_curve = self.head_at_speed(self.Q_rated, self.N_rated)
        if abs(H_curve - self.H_rated) > self.H_rated * 0.1:
            # Auto-adjust constant to match rated point
            self.h_c = self.H_rated - self.h_a * self.Q_rated**2 - self.h_b * self.Q_rated
    
    def head(self, Q: float, N: Optional[float] = None) -> float:
        """
        Head at flow Q and speed N (affinity laws).
        
        Parameters
        ----------
        Q : float
            Flow rate (m³/s)
        N : float, optional
            Pump speed (RPM). Defaults to rated speed.
            
        Returns
        -------
        float
            Head (m)
        """
        if N is None:
            N = self.N_rated
        
        speed_ratio = N / self.N_rated
        
        # Head at rated speed
        H_rated = self.h_a * Q**2 + self.h_b * Q + self.h_c
        
        # Apply affinity law
        return H_rated * speed_ratio**2
    
    def head_at_speed(self, Q: float, N: float) -> float:
        """Head at a specific speed (uses affinity)."""
        return self.head(Q, N)
    
    def efficiency(self, Q: float) -> float:
        """Hydraulic efficiency at flow Q."""
        eff = self.e_a * Q**2 + self.e_b * Q + self.e_c
        return max(0.0, min(1.0, eff))
    
    def power(self, Q: float, N: Optional[float] = None) -> float:
        """
        Shaft power requirement (kW).
        
        P = ρ·g·Q·H / η
        """
        if N is None:
            N = self.N_rated
        
        H = self.head(Q, N)
        eff = self.efficiency(Q)
        
        rho = self.fluid_density
        g = 9.81
        
        if eff < 0.01:
            return 0.0
        
        P_watts = rho * g * Q * H / eff
        return P_watts / 1000.0  # kW
    
    def npsh_required(self, Q: float) -> float:
        """
        Required NPSH (Net Positive Suction Head) in meters.
        Typically NPSHr increases with Q².
        """
        q_ratio = Q / max(self.Q_rated, 0.001)
        return 3.0 + 1.5 * q_ratio**2
    
    def with_speed(self, speed_ratio: float) -> 'Pump':
        """
        Create a pump simulation with a different speed ratio.
        
        Returns a Pump with modified head curve for the new speed.
        """
        N_new = self.N_rated * speed_ratio
        new_pump = Pump(
            name=f"{self.name} @ {N_new:.0f} RPM",
            Q_rated=self.Q_rated * speed_ratio,
            H_rated=self.H_rated * speed_ratio**2,
            N_rated=N_new,
            efficiency_rated=self.efficiency_rated,
            h_a=self.h_a * speed_ratio**2,
            h_b=self.h_b * speed_ratio,
            h_c=self.h_c * speed_ratio**2,
            e_a=self.e_a, e_b=self.e_b, e_c=self.e_c,
            fluid_density=self.fluid_density,
        )
        return new_pump
    
    def operating_point(
        self, 
        system_head_func: Callable[[float], float],
        N: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """
        Find pump operating point (Q, H, P) given system resistance curve.
        
        System head: H_sys(Q) = H_static + k·Q²
        
        Returns (Q, H, Power_kW) at the intersection of pump curve
        and system curve.
        """
        if N is None:
            N = self.N_rated
        
        # Binary search for Q where pump H = system H
        Q_min, Q_max = 0.01, self.Q_rated * 2.0
        
        for _ in range(50):
            Q_mid = (Q_min + Q_max) / 2.0
            H_pump = self.head(Q_mid, N)
            H_sys = system_head_func(Q_mid)
            
            if H_pump > H_sys:
                Q_min = Q_mid
            else:
                Q_max = Q_mid
        
        Q_op = Q_min
        H_op = self.head(Q_op, N)
        P_op = self.power(Q_op, N)
        
        return Q_op, H_op, P_op


def pump_combo_series(pumps: list, Q: float) -> Tuple[float, float]:
    """
    Pumps in series: heads add, flow same.
    Returns (total_head, total_power_kW).
    """
    total_H = sum(p.head(Q) for p in pumps)
    total_P = sum(p.power(Q) for p in pumps)
    return total_H, total_P


def pump_combo_parallel(pumps: list, Q_total: float) -> Tuple[float, float]:
    """
    Pumps in parallel: flows add, head same.
    Each pump handles Q_total / N.
    Returns (head, total_power_kW).
    """
    N = len(pumps)
    Q_per_pump = Q_total / N
    if any(p.max_flow and Q_per_pump > p.max_flow for p in pumps if hasattr(p, 'max_flow')):
        raise ValueError("Flow per pump exceeds maximum")
    
    H = pumps[0].head(Q_per_pump)
    total_P = sum(p.power(Q_per_pump) for p in pumps)
    return H, total_P


# ============================================================
# Demo: Pump analysis
# ============================================================
def demo_pump():
    """Quick pump analysis demo."""
    pump = Pump(name="Main Pipeline Pump", Q_rated=0.3, H_rated=480, N_rated=1500)
    
    print("=" * 60)
    print("Pump Analysis")
    print("=" * 60)
    
    # Test different flow rates
    print(f"\n{'Q (L/s)':>10} {'H (m)':>10} {'η':>8} {'Power (kW)':>12}")
    print("-" * 42)
    for Q in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
        H = pump.head(Q)
        eff = pump.efficiency(Q)
        P = pump.power(Q)
        print(f"{Q*1000:>8.0f}  {H:>8.1f}  {eff:>7.1%}  {P:>10.1f}")
    
    # Variable speed
    print(f"\nVariable Speed (Q={pump.Q_rated*1000:.0f} L/s):")
    print(f"{'Speed (%)':>10} {'N (RPM)':>10} {'H (m)':>10} {'Power (kW)':>12}")
    print("-" * 44)
    for speed_pct in [60, 80, 100, 110, 120]:
        N = pump.N_rated * speed_pct / 100
        H = pump.head(pump.Q_rated, N)
        P = pump.power(pump.Q_rated, N)
        print(f"{speed_pct:>8.0f}%  {N:>8.0f}  {H:>8.1f}  {P:>10.1f}")


if __name__ == "__main__":
    demo_pump()
