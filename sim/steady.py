"""
Steady-state pipeline analysis module.

Computes steady-state pressure and temperature profiles analytically,
providing proper initialization for transient simulations.

Key insight: For liquid pipelines, the thermal time scale (hours) 
is orders of magnitude larger than the hydraulic time scale (seconds).
A proper steady-state initialization is essential for realistic 
transient results.
"""

import numpy as np
from typing import Tuple, Callable, Optional
from .fluid import Liquid
from .pipe import Pipe
from .solver import SinglePhaseTransientSolver


class SteadyStateCalculator:
    """
    Computes steady-state pressure and temperature profiles along a pipeline.
    
    Provides proper initialization states for transient simulations.
    """
    
    def __init__(self, pipe: Pipe, liquid: Liquid):
        self.pipe = pipe
        self.liquid = liquid
        self.g = 9.81  # m/s²
    
    def pressure_profile(
        self, 
        Q: float, 
        P_outlet: float, 
        N: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute steady-state pressure along the pipeline.
        
        P(x) = P_outlet + ρg·(z_L - z(x)) + f·ρ·(L-x)·V²/(2D)
        
        Where elevation is handled via interpolation.
        
        Parameters
        ----------
        Q : float
            Flow rate (m³/s)
        P_outlet : float
            Outlet pressure (Pa)
        N : int
            Number of grid points
            
        Returns
        -------
        x : np.ndarray
            Distance from inlet (m)
        P : np.ndarray
            Pressure at each point (Pa)
        """
        A = self.pipe.area()
        V = Q / A
        D = self.pipe.diameter
        rho = self.liquid.rho_ref
        f = self._friction_factor(V, D, rho)
        
        x = np.linspace(0, self.pipe.length, N)
        
        # Elevation at each point
        z = np.interp(x, [0, self.pipe.length], 
                      [self.pipe.elevation_start, self.pipe.elevation_end])
        
        # Pressure: outlet reference + elevation + friction
        dP_friction = f * (self.pipe.length - x) / D * rho * V**2 / 2
        dP_elevation = rho * self.g * (self.pipe.elevation_end - z)
        
        P = P_outlet + dP_elevation + dP_friction
        
        return x, P
    
    def temperature_profile(
        self,
        Q: float,
        T_inlet: float,
        N: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute steady-state temperature along the pipeline.
        
        Analytical solution of steady energy equation:
            dT/dx = A - B·(T - T_ground)
        
        where:
            A = f·V³/(2·D·cp)           (friction heating)
            B = 4·U/(D·ρ·cp·V)          (heat loss coefficient)
        
        Analytical solution:
            T(x) = T_ground + A/B + (T_inlet - T_ground - A/B)·exp(-B·x)
        
        Parameters
        ----------
        Q : float
            Flow rate (m³/s)
        T_inlet : float
            Inlet temperature (°C)
        N : int
            Number of grid points
            
        Returns
        -------
        x : np.ndarray
            Distance from inlet (m)
        T : np.ndarray
            Temperature at each point (°C)
        """
        A = self.pipe.area()
        V = Q / A
        D = self.pipe.diameter
        rho = self.liquid.rho_ref
        cp = self.liquid.cp
        U = self.pipe.heat_transfer_coeff
        T0 = self.pipe.T_ground
        
        f = self._friction_factor(V, D, rho)
        
        # Friction heating term
        A_term = f * V**3 / (2 * D * cp)
        # Heat loss coefficient
        B_term = 4 * U / (D * rho * cp * abs(V) + 1e-10)
        
        x = np.linspace(0, self.pipe.length, N)
        
        if B_term * self.pipe.length > 30:
            # Thermally fully developed before outlet
            T = T0 + A_term / B_term + (T_inlet - T0 - A_term / B_term) * np.exp(-B_term * x)
        else:
            # General solution
            T = T0 + A_term / B_term + (T_inlet - T0 - A_term / B_term) * np.exp(-B_term * x)
        
        return x, T
    
    def initialize_transient(
        self,
        Q: float,
        T_inlet: float,
        P_outlet: float,
        solver: SinglePhaseTransientSolver,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Create proper initial conditions for a transient simulation.
        
        Returns initial velocity, pressure array, temperature array.
        """
        A = self.pipe.area()
        V0 = Q / A
        
        # Steady pressure
        x_P, P = self.pressure_profile(Q, P_outlet, solver.Nx + 1)
        
        # Steady temperature
        x_T, T = self.temperature_profile(Q, T_inlet, solver.Nx + 1)
        
        return V0, P, T
    
    def _friction_factor(self, V: float, D: float, rho: float) -> float:
        """Darcy-Weisbach friction factor (Colebrook-White)."""
        Re = rho * V * D / self.liquid.viscosity_ref if V > 0 else 1e3
        eps = self.pipe.roughness
        
        if Re < 2000:
            return 64 / max(Re, 100)  # Laminar
        elif Re < 4000:
            f_lam = 64 / Re
            f_turb = self._colebrook(Re, eps / D)
            # Smooth transition
            frac = (Re - 2000) / 2000
            return f_lam * (1 - frac) + f_turb * frac
        else:
            return self._colebrook(Re, eps / D)
    
    def _colebrook(self, Re: float, rel_rough: float) -> float:
        """Colebrook-White friction factor (Haaland approximation)."""
        if Re <= 0:
            return 0.02
        from math import log10
        try:
            return 0.25 / (log10(rel_rough / 3.7 + 5.74 / Re**0.9))**2
        except (ValueError, ZeroDivisionError):
            return 0.02


def analyze_pipeline(
    length: float = 15000.0,
    diameter: float = 0.6,
    Q: float = 0.25,
    P_outlet: float = 0.5e6,
    T_inlet: float = 45.0,
    liquid_type: str = "crude_oil",
    verbose: bool = True,
):
    """
    Quick pipeline analysis tool - report key operating parameters.
    """
    from .pipe import Pipe
    from .fluid import Liquid
    
    # Create pipe
    pipe = Pipe(length=length, diameter=diameter, wall_thickness=0.014,
                roughness=4.5e-5, elevation_start=0.0, elevation_end=50.0)
    
    # Create fluid
    if liquid_type == "crude_oil":
        liquid = Liquid(name="Crude Oil", rho_ref=860.0, bulk_modulus=1.8e9,
                        cp=2000.0, viscosity_ref=12e-3, thermal_expansion=8.0e-4)
    elif liquid_type == "water":
        liquid = Liquid()
    else:
        raise ValueError(f"Unknown liquid type: {liquid_type}")
    
    calc = SteadyStateCalculator(pipe, liquid)
    A = pipe.area()
    V = Q / A
    f = calc._friction_factor(V, pipe.diameter, liquid.rho_ref)
    Re = liquid.rho_ref * V * pipe.diameter / liquid.viscosity_ref
    
    dP_friction = f * pipe.length / pipe.diameter * liquid.rho_ref * V**2 / 2
    dP_elevation = liquid.rho_ref * 9.81 * (pipe.elevation_end - pipe.elevation_start)
    dP_total = dP_friction + dP_elevation
    P_inlet = P_outlet + dP_total
    
    x_T, T_profile = calc.temperature_profile(Q, T_inlet, 50)
    temp_drop = T_inlet - T_profile[-1]
    
    if verbose:
        print(f"📊 Pipeline Analysis Report")
        print(f"{'='*60}")
        print(f"  Fluid: {liquid.name}")
        print(f"  Pipe: L={length/1000:.1f}km, D={diameter*1000:.0f}mm")
        print(f"  Flow: Q={Q*1000:.0f} L/s, V={V:.2f} m/s")
        print(f"  Re={Re:.0f}, f={f:.4f}")
        print(f"  Pressure drop: {dP_total/1e6:.3f} MPa")
        print(f"    - Friction: {dP_friction/1e6:.3f} MPa")
        print(f"    - Elevation: {dP_elevation/1e6:.3f} MPa")
        print(f"  P_inlet={P_inlet/1e6:.3f} MPa, P_outlet={P_outlet/1e6:.3f} MPa")
        print(f"  T_inlet={T_inlet:.1f}°C, T_outlet={T_profile[-1]:.1f}°C")
        print(f"  Temp drop: {temp_drop:.1f}°C (over {length/1000:.0f}km)")
        print(f"  Thermal time scale: L/V={length/V/3600:.1f} hours")
    
    return {
        "pipe": pipe,
        "liquid": liquid,
        "V": V,
        "Re": Re,
        "f": f,
        "P_inlet": P_inlet,
        "P_outlet": P_outlet,
        "dP": dP_total,
        "T_inlet": T_inlet,
        "T_outlet": T_profile[-1],
        "T_profile": T_profile,
        "x_T": x_T,
    }
