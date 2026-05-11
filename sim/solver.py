"""
solver.py — Coupled hydraulic-thermal transient solver for single-phase pipelines
Method of Characteristics (hydraulics) + Upwind Finite Difference (temperature)
"""
import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple, Optional

from .fluid import Liquid
from .pipe import Pipe

# ============================================================
# Results container
# ============================================================

@dataclass
class TransientResult:
    """Simulation results"""
    t: np.ndarray          # time array (s)
    x: np.ndarray          # distance array (m)
    P: np.ndarray          # pressure (Pa)  [Nt x Nx]
    T: np.ndarray          # temperature (°C) [Nt x Nx]
    V: np.ndarray          # velocity (m/s) [Nt x Nx]
    Q: np.ndarray          # flow rate (m³/s) [Nt x Nx]
    rho: np.ndarray        # density (kg/m³) [Nt x Nx]
    mu: np.ndarray         # viscosity (Pa·s) [Nt x Nx]
    
    @property
    def Nt(self) -> int:
        return len(self.t)
    
    @property
    def Nx(self) -> int:
        return len(self.x)
    
    def at(self, i_node: int):
        """Extract data at a specific node"""
        return {
            't': self.t,
            'P': self.P[:, i_node],
            'T': self.T[:, i_node],
            'V': self.V[:, i_node],
            'Q': self.Q[:, i_node],
        }


# ============================================================
# Solver
# ============================================================

class SinglePhaseTransientSolver:
    """
    Coupled hydraulic-thermal transient solver for single-phase liquid pipelines.
    
    Governing equations:
    - Continuity + Momentum → solved by MOC for head H and velocity V
    - Energy → solved by upwind finite difference for temperature T
    - Equation of state → ρ = ρ(P, T)
    
    Boundary conditions (choose one of two modes):
    Mode A: Inlet Q(T), T_in(t)  |  Outlet P_out(t)
    Mode B: Inlet P_in(t), T_in(t)  |  Outlet Q_out(t)
    """
    
    def __init__(self, pipe: Pipe, liquid: Liquid, Nx: int = 20):
        self.pipe = pipe
        self.liquid = liquid
        self.a = pipe.wave_speed(liquid)
        self.Nx = Nx
        self.dx = pipe.length / Nx
        self.dt = self.dx / self.a  # CFL condition
        self.g = 9.81
        
        # Array of spatial positions
        self.x = np.linspace(0, pipe.length, Nx + 1)
        self.elev = pipe.elevation(self.x)
        
        print(f"  Wave speed: {self.a:.1f} m/s")
        print(f"  Grid: Nx={Nx}, dx={self.dx:.2f}m, dt={self.dt:.4f}s")
    
    def _init_arrays(self, Nt: int):
        """Initialize solution arrays"""
        return (
            np.zeros((Nt, self.Nx + 1)),  # H head
            np.zeros((Nt, self.Nx + 1)),  # V velocity
            np.zeros((Nt, self.Nx + 1)),  # T temperature
            np.zeros((Nt, self.Nx + 1)),  # rho density
            np.zeros((Nt, self.Nx + 1)),  # mu viscosity
        )
    
    def _init_steady(
        self, H0: np.ndarray, V0: np.ndarray, T0: np.ndarray
    ) -> Tuple[np.ndarray, ...]:
        """
        Initialize from given steady-state profiles
        Or create simple uniform initial condition
        """
        Nt = len(H0) if hasattr(H0, '__len__') else 1
        H_arr, V_arr, T_arr, rho_arr, mu_arr = self._init_arrays(1)
        
        H_arr[0] = H0 if isinstance(H0, (int, float)) else H0
        V_arr[0] = V0 if isinstance(V0, (int, float)) else V0
        T_arr[0] = T0 if isinstance(T0, (int, float)) else T0
        
        # Compute initial P from H
        P0 = self.liquid.density(
            np.full_like(self.x, 1e5),
            T_arr[0]
        ) * self.g * H_arr[0]
        
        rho_arr[0] = self.liquid.density(P0, T_arr[0])
        mu_arr[0] = self.liquid.viscosity(T_arr[0])
        
        return H_arr[0], V_arr[0], T_arr[0], rho_arr[0], mu_arr[0]
    
    def solve(
        self,
        t_max: float,
        inlet_bc: Callable,   # returns (Q, T) or (P, T) depending on mode
        outlet_bc: Callable,  # returns P or Q depending on mode
        mode: str = 'A',      # 'A': inlet Q+T / outlet P  |  'B': inlet P+T / outlet Q
        H_initial: float = 50.0,
        V_initial: float = 1.0,
        T_initial: float = 20.0,
        P_initial: Optional[float] = None,
    ) -> TransientResult:
        """
        Solve coupled hydraulic-thermal transient
        
        mode='A': Inlet specifies flow rate (m³/s) + temperature (°C)
                  Outlet specifies pressure (Pa)
        mode='B': Inlet specifies pressure (Pa) + temperature (°C)
                  Outlet specifies flow rate (m³/s)
        """
        Nt = int(t_max / self.dt) + 1
        
        # Initialize arrays
        H = np.zeros((Nt, self.Nx + 1))
        V = np.zeros((Nt, self.Nx + 1))
        T = np.zeros((Nt, self.Nx + 1))
        rho = np.zeros((Nt, self.Nx + 1))
        mu_arr = np.zeros((Nt, self.Nx + 1))
        
        # Reference pressure for density calculation
        P_ref_arr = np.full(self.Nx + 1, self.liquid.P_ref)
        
        # Initial state
        if P_initial is not None:
            # User specified initial pressure - compute equivalent head
            rho0 = self.liquid.density(
                np.full(self.Nx + 1, P_initial),
                np.full(self.Nx + 1, T_initial)
            )
            H0 = P_initial / (rho0 * self.g) + self.elev  # head including elevation
        else:
            H0 = H_initial + self.elev
        
        V0_arr = np.full(self.Nx + 1, V_initial)
        T0_arr = np.full(self.Nx + 1, T_initial)
        
        H[0] = H0
        V[0] = V0_arr
        T[0] = T0_arr
        
        # Compute initial density and viscosity
        if P_initial is not None:
            P_init_arr = np.full(self.Nx + 1, P_initial)
        else:
            P_init_arr = self.liquid.rho_ref * self.g * H[0]
        rho[0] = self.liquid.density(P_init_arr, T[0])
        mu_arr[0] = self.liquid.viscosity(T[0])
        
        # Wave speed and friction parameters
        B = self.a / self.g  # characteristic impedance coefficient
        
        # ===== Time marching =====
        for n in range(Nt - 1):
            t = n * self.dt
            
            # Compute friction at current state
            A_pipe = self.pipe.area()
            Re = np.maximum(
                rho[n] * np.abs(V[n]) * self.pipe.diameter / np.maximum(mu_arr[n], 1e-10),
                1.0
            )
            f = self.pipe.friction_factor(V[n], Re)
            
            # Friction damping term for MOC
            R = f * self.dx / (2 * self.g * self.pipe.diameter) * np.ones(self.Nx + 1)
            
            # Elevation term
            sin_theta = (self.elev[-1] - self.elev[0]) / self.pipe.length
            elev_term = sin_theta * self.dx / self.a * self.g
            
            # ============ Hydraulic solve (MOC) ============
            
            # --- Internal nodes ---
            for i in range(1, self.Nx):
                # C⁺: from i-1 → i, along dx/dt = +a
                q_abs = abs(V[n, i-1])
                Cp = (H[n, i-1] + B * V[n, i-1] 
                      - R[i-1] * V[n, i-1] * q_abs
                      - elev_term * V[n, i-1] / abs(V[n, i-1] + 1e-10)
                      - self.elev[i-1] + self.elev[i])
                
                # C⁻: from i+1 → i, along dx/dt = -a
                q_abs = abs(V[n, i+1])
                Cm = (H[n, i+1] - B * V[n, i+1] 
                      + R[i+1] * V[n, i+1] * q_abs
                      + elev_term * V[n, i+1] / abs(V[n, i+1] + 1e-10)
                      - self.elev[i+1] + self.elev[i])
                
                V[n+1, i] = 0.5 * (Cp - Cm) / B
                H[n+1, i] = 0.5 * (Cp + Cm)
            
            # ============ Boundary conditions ============
            
            if mode.upper() == 'A':
                # Mode A: Inlet Q+T / Outlet P
                
                # ---- Upstream (x=0): specify flow rate + temperature ----
                Q_in, T_in = inlet_bc(t)
                V_in = Q_in / A_pipe
                
                # Use C⁻ characteristic from node 1
                q_abs = abs(V[n, 1])
                Cm0 = (H[n, 1] - B * V[n, 1] 
                       + R[1] * V[n, 1] * q_abs
                       + elev_term * V[n, 1] / abs(V[n, 1] + 1e-10)
                       - self.elev[1] + self.elev[0])
                
                V[n+1, 0] = V_in
                H[n+1, 0] = Cm0 + B * V_in
                
                # Temperature at inlet is specified
                T[n+1, 0] = T_in
                
                # ---- Downstream (x=L): specify pressure ----
                P_out = outlet_bc(t)
                
                # Use C⁺ characteristic from node Nx-1
                q_abs = abs(V[n, self.Nx-1])
                CpL = (H[n, self.Nx-1] + B * V[n, self.Nx-1] 
                       - R[self.Nx-1] * V[n, self.Nx-1] * q_abs
                       - elev_term * V[n, self.Nx-1] / abs(V[n, self.Nx-1] + 1e-10)
                       - self.elev[self.Nx-1] + self.elev[self.Nx])
                
                # Convert P to head
                rho_out = rho[n, self.Nx]
                H_out = P_out / (rho_out * self.g)
                
                H[n+1, self.Nx] = H_out
                V[n+1, self.Nx] = (CpL - H_out) / B
                
            elif mode.upper() == 'B':
                # Mode B: Inlet P+T / Outlet Q
                
                # ---- Upstream (x=0): specify pressure + temperature ----
                P_in, T_in = inlet_bc(t)
                rho_in = rho[n, 0]
                H_in = P_in / (rho_in * self.g)
                
                # Use C⁻ characteristic
                q_abs = abs(V[n, 1])
                Cm0 = (H[n, 1] - B * V[n, 1] 
                       + R[1] * V[n, 1] * q_abs
                       + elev_term * V[n, 1] / abs(V[n, 1] + 1e-10)
                       - self.elev[1] + self.elev[0])
                
                H[n+1, 0] = H_in
                V[n+1, 0] = (H_in - Cm0) / B
                T[n+1, 0] = T_in
                
                # ---- Downstream (x=L): specify flow rate ----
                Q_out = outlet_bc(t)
                V_out = Q_out / A_pipe
                
                q_abs = abs(V[n, self.Nx-1])
                CpL = (H[n, self.Nx-1] + B * V[n, self.Nx-1] 
                       - R[self.Nx-1] * V[n, self.Nx-1] * q_abs
                       - elev_term * V[n, self.Nx-1] / abs(V[n, self.Nx-1] + 1e-10)
                       - self.elev[self.Nx-1] + self.elev[self.Nx])
                
                V[n+1, self.Nx] = V_out
                H[n+1, self.Nx] = CpL - B * V_out
            
            # ============ Temperature solve (Energy equation) ============
            
            # Energy equation (upwind finite difference):
            # ρ·cp·(∂T/∂t + V·∂T/∂x) = f·ρ·V³/(2D) - 4U·(T-T₀)/D
            # 
            # Discretization (explicit upwind):
            # For V > 0: backward difference in space
            # For V < 0: forward difference in space
            
            cp = self.liquid.cp
            U = self.pipe.heat_transfer_coeff
            D = self.pipe.diameter
            T_g = self.pipe.T_ground
            
            # Internal nodes
            for i in range(1, self.Nx):
                Vi = V[n+1, i]
                rho_i = rho[n, i]
                fi = f[i]
                
                # Friction heating
                q_friction = fi * rho_i * abs(Vi)**3 / (2 * D)
                
                # Heat loss to ground
                q_loss = 4.0 * U * (T[n, i] - T_g) / D
                
                # Advection: upwind
                if Vi >= 0:
                    dTdx = (T[n, i] - T[n, i-1]) / self.dx
                else:
                    dTdx = (T[n, i+1] - T[n, i]) / self.dx
                
                # Temperature update (explicit Euler)
                T_new = (T[n, i] 
                         + self.dt * (-Vi * dTdx 
                                      + (q_friction - q_loss) / (rho_i * cp)))
                # Clip temperature to physically valid range
                T[n+1, i] = max(-50.0, min(500.0, T_new))
            
            # Downstream temperature (zero-gradient or extrapolation)
            if V[n+1, self.Nx] >= 0:
                T[n+1, self.Nx] = T[n+1, self.Nx-1]
            else:
                T[n+1, self.Nx] = T[n, self.Nx]
            
            # ============ Update fluid properties ============
            P_curr = rho[n] * self.g * H[n+1]
            rho[n+1] = self.liquid.density(P_curr, T[n+1])
            mu_arr[n+1] = self.liquid.viscosity(T[n+1])
        
        # ===== Build result =====
        P = np.zeros_like(H)
        Q = np.zeros_like(H)
        
        for n in range(Nt):
            for i in range(self.Nx + 1):
                P[n, i] = rho[n, i] * self.g * (H[n, i] - self.elev[i])
                Q[n, i] = V[n, i] * A_pipe
        
        return TransientResult(
            t=np.arange(Nt) * self.dt,
            x=self.x,
            P=P, T=T, V=V, Q=Q,
            rho=rho, mu=mu_arr,
        )


# ============================================================
# Boundary condition factories
# ============================================================

def flow_inlet(Q_func: Callable[[float], float], T_func: Callable[[float], float]):
    """Inlet boundary: flow rate + temperature (Mode A)"""
    def bc(t: float):
        return Q_func(t), T_func(t)
    return bc

def pressure_inlet(P_func: Callable[[float], float], T_func: Callable[[float], float]):
    """Inlet boundary: pressure + temperature (Mode B)"""
    def bc(t: float):
        return P_func(t), T_func(t)
    return bc

def pressure_outlet(P_func: Callable[[float], float]):
    """Outlet boundary: pressure (Mode A)"""
    return P_func

def flow_outlet(Q_func: Callable[[float], float]):
    """Outlet boundary: flow rate (Mode B)"""
    return Q_func


# ============================================================
# Utility: step/ramp functions for transients
# ============================================================

def step(value_initial: float, value_final: float, t_start: float, duration: float = 0.0):
    """Step or ramp change"""
    if duration <= 0:
        return lambda t: value_final if t >= t_start else value_initial
    else:
        def ramp(t):
            if t <= t_start:
                return value_initial
            elif t >= t_start + duration:
                return value_final
            else:
                frac = (t - t_start) / duration
                return value_initial + (value_final - value_initial) * frac
        return ramp
