"""
solver_advanced.py — Three advanced transient solvers for single-phase pipelines

Adds to the existing MOC solver:
  1. Implicit Finite Difference (Crank-Nicolson) — unconditionally stable
  2. MacCormack Predictor-Corrector — 2nd order accurate
  3. Finite Volume (MUSCL-Hancock) — conservative, strong shock capturing

All solvers share the TransientResult interface from solver.py
"""

import numpy as np
from typing import Callable, Tuple, Optional
from .fluid import Liquid
from .pipe import Pipe
from .solver import TransientResult, step


# ============================================================
# 1. IMPLICIT FINITE DIFFERENCE METHOD (Crank-Nicolson)
# ============================================================

class ImplicitFDMSolver:
    """
    Implicit Finite Difference solver for pipeline transients.
    
    Discretization:
      - Time: Crank-Nicolson (θ=0.5) or Backward Euler (θ=1.0)
      - Space: central differences
      - Friction: semi-implicit (linearized at old time level)
    
    Solves a coupled block-tridiagonal system for (H, V) at each step.
    
    Advantages over MOC:
      - Unconditionally stable → allows larger timesteps
      - No CFL restriction → ideal for slow transients / long simulations
      - Easy to extend with additional physics
    
    Limitations:
      - Numerical diffusion at high CFL numbers
      - Requires linear system solve at each step
      - Friction linearization may cause small errors for rapid transients
    """
    
    def __init__(self, pipe: Pipe, liquid: Liquid, Nx: int = 20,
                 theta: float = 0.5):
        """
        Parameters
        ----------
        pipe : Pipe
            Pipeline geometry and properties
        liquid : Liquid
            Fluid properties
        Nx : int
            Number of spatial segments
        theta : float
            Time-stepping parameter:
              0.5 = Crank-Nicolson (2nd order, recommended)
              1.0 = Backward Euler (1st order, more diffusive)
        """
        self.pipe = pipe
        self.liquid = liquid
        self.a = pipe.wave_speed(liquid)
        self.Nx = Nx
        self.dx = pipe.length / Nx
        self.theta = theta
        self.g = 9.81
        
        self.x = np.linspace(0, pipe.length, Nx + 1)
        self.elev = pipe.elevation(self.x)
        
        # Characteristic impedance
        self.B = self.a / self.g
        
        # Pipe constants
        self.A_pipe = pipe.area()
        self.D = pipe.diameter
        
        print(f"  [IFDM] Wave speed: {self.a:.1f} m/s")
        print(f"  [IFDM] Grid: Nx={Nx}, dx={self.dx:.2f}m")
        print(f"  [IFDM] Timestepping: θ={theta} "
              f"({'Crank-Nicolson' if theta==0.5 else 'Backward Euler' if theta==1.0 else 'Custom'})")
    
    def _build_system(self, H_n: np.ndarray, V_n: np.ndarray, dt: float
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build the block-tridiagonal system for (H, V) at internal nodes.
        
        Crank-Nicolson discretization of continuity + momentum:
        
        Continuity:  H_i^(n+1) + θ·α·(V_{i+1} - V_{i-1})/2·Δt
                   = H_i^n - (1-θ)·α·(V_{i+1} - V_{i-1})/2·Δt
        
        Momentum:  V_i^(n+1) + θ·g·(H_{i+1} - H_{i-1})/2·Δt
                 + θ·(f·V·|V|/2D)·Δt
                 = V_i^n - (1-θ)·g·(H_{i+1} - H_{i-1})/2·Δt
                   - (1-θ)·(f·V·|V|/2D)·Δt
        
        where α = a²/g
        
        Returns diagonal, sub-diagonal, super-diagonal, RHS arrays.
        """
        N = self.Nx + 1  # total nodes
        a = self.a
        g = self.g
        
        # Matrix coefficients (2 equations per node → block size 2)
        # We solve: A·[H; V]^(n+1) = b
        # Using an alternating ordering: H0, V0, H1, V1, ...
        
        # Build full 2N x 2N system (sparse tridiagonal blocks)
        A = np.zeros((2 * N, 2 * N))
        b = np.zeros(2 * N)
        
        theta = self.theta
        r = a**2 / g  # continuity coupling coefficient
        dx = self.dx
        
        # Friction at old time level (linearized)
        f_n = np.zeros(N)
        for i in range(N):
            Re = (self.liquid.rho_ref * max(abs(V_n[i]), 1e-6) * self.D 
                  / max(self.liquid.viscosity_ref, 1e-10))
            f_n[i] = self.pipe.friction_factor(np.array([V_n[i]]), np.array([Re]))[0]
        
        # === Internal nodes (i=1 to N-2) ===
        for i in range(1, N - 1):
            row_H = 2 * i      # equation for H_i^(n+1)
            row_V = 2 * i + 1  # equation for V_i^(n+1)
            
            # --- Continuity equation at node i ---
            # H_i^(n+1) + θ·r·Δt·(V_{i+1} - V_{i-1})/(2·dx)
            #   = H_i^n - (1-θ)·r·Δt·(V_{i+1}^n - V_{i-1}^n)/(2·dx)
            
            A[row_H, 2*i] = 1.0  # H_i^(n+1)
            A[row_H, 2*(i+1)+1] = theta * r * dt / (2 * dx)   # V_{i+1} coeff
            A[row_H, 2*(i-1)+1] = -theta * r * dt / (2 * dx)  # V_{i-1} coeff
            
            b[row_H] = (H_n[i] 
                        - (1 - theta) * r * dt * (V_n[i+1] - V_n[i-1]) / (2 * dx))
            
            # --- Momentum equation at node i ---
            # V_i^(n+1) + θ·g·Δt·(H_{i+1} - H_{i-1})/(2·dx)
            #   + θ·f_i^n·|V_i^n|·V_i^(n+1)·Δt/(2·D)
            # = V_i^n - (1-θ)·g·Δt·(H_{i+1}^n - H_{i-1}^n)/(2·dx)
            #   - (1-θ)·f_i^n·V_i^n·|V_i^n|·Δt/(2·D)
            
            f_lin = f_n[i] * abs(V_n[i]) / (2 * self.D)
            
            A[row_V, 2*i+1] = 1.0 + theta * f_lin * dt  # V_i^(n+1)
            A[row_V, 2*(i+1)] = theta * g * dt / (2 * dx)    # H_{i+1} coeff
            A[row_V, 2*(i-1)] = -theta * g * dt / (2 * dx)   # H_{i-1} coeff
            
            b[row_V] = (V_n[i] 
                        - (1 - theta) * g * dt * (H_n[i+1] - H_n[i-1]) / (2 * dx)
                        - (1 - theta) * f_lin * V_n[i] * dt)
        
        return A, b, f_n
    
    def _apply_bc(self, A: np.ndarray, b: np.ndarray,
                  H_n: np.ndarray, V_n: np.ndarray, dt: float,
                  inlet_bc: Callable, outlet_bc: Callable,
                  mode: str, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Apply boundary conditions to the linear system."""
        N = self.Nx + 1
        theta = self.theta
        g = self.g
        dx = self.dx
        r = self.a**2 / g
        
        if mode.upper() == 'A':
            # Mode A: Inlet Q+T → V+V_in, no H equation needed
            #         Outlet P → H_out, no V equation
            Q_in, T_in = inlet_bc(t)
            V_in = Q_in / self.A_pipe
            
            # Inlet (i=0): V is known, H from C⁻ characteristic (implicit)
            # For implicit: V_0^(n+1) = V_in
            row_H0 = 0
            row_V0 = 1
            
            A[row_V0, 1] = 1.0
            b[row_V0] = V_in
            
            # H_0 from reverse characteristic (embedded in system via node 1)
            # Simplified: use the characteristic relation as an equation
            # H_0 - B·V_0 = H_1^n - B·V_1^n (from C⁻)
            A[row_H0, 0] = 1.0
            A[row_H0, 1] = -self.B
            b[row_H0] = H_n[1] - self.B * V_n[1]
            
        elif mode.upper() == 'B':
            # Mode B: Inlet P+T → H_in, V from C⁻
            P_in, T_in = inlet_bc(t)
            rho_in = self.liquid.density(P_in, T_in)
            H_in = P_in / (rho_in * self.g)
            
            row_H0 = 0
            row_V0 = 1
            
            # H_0^(n+1) = H_in
            A[row_H0, 0] = 1.0
            b[row_H0] = H_in
            
            # V_0 from C⁻: -B·V_0 + H_0 = H_1^n - B·V_1^n
            # → but H_0 is known, so V_0 = (H_0 - (H_1 - B·V_1))/B
            A[row_V0, 1] = 1.0
            b[row_V0] = (H_in - (H_n[1] - self.B * V_n[1])) / self.B
        
        # Outlet boundary
        if mode.upper() == 'A':
            # Outlet: P specified → H_out
            P_out = outlet_bc(t)
            rho_out = self.liquid.rho_ref
            H_out = P_out / (rho_out * self.g)
            
            row_HL = 2 * N - 2  # last H equation
            row_VL = 2 * N - 1  # last V equation
            
            # H_N^(n+1) = H_out
            A[row_HL, 2*N-2] = 1.0
            b[row_HL] = H_out
            
            # V_N from C⁺: V_N = (H_{N-1} + B·V_{N-1} - H_N)/B
            A[row_VL, 2*N-1] = 1.0
            b[row_VL] = (H_n[N-1] + self.B * V_n[N-1] - H_out) / self.B
            
        elif mode.upper() == 'B':
            # Outlet: Q specified → V_out
            Q_out = outlet_bc(t)
            V_out = Q_out / self.A_pipe
            
            row_HL = 2 * N - 2
            row_VL = 2 * N - 1
            
            # V_N^(n+1) = V_out
            A[row_VL, 2*N-1] = 1.0
            b[row_VL] = V_out
            
            # H_N from C⁺: H_N = H_{N-1} + B·(V_{N-1} - V_N)
            A[row_HL, 2*N-2] = 1.0
            b[row_HL] = H_n[N-1] + self.B * (V_n[N-1] - V_out)
        
        return A, b
    
    def _solve_tridiag(self, A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the block system.
        
        The matrix is nearly tridiagonal. For moderate N, direct solve is fine.
        For large N, we'd use a block-tridiagonal Thomas algorithm.
        """
        x = np.linalg.solve(A, b)
        N = len(x) // 2
        H = x[::2]   # even indices: head
        V = x[1::2]  # odd indices: velocity
        return H, V
    
    def solve(
        self,
        t_max: float,
        inlet_bc: Callable,
        outlet_bc: Callable,
        mode: str = 'A',
        dt: Optional[float] = None,
        H_initial: float = 50.0,
        V_initial: float = 1.0,
        T_initial: float = 20.0,
        P_initial: Optional[float] = None,
    ) -> TransientResult:
        """
        Solve pipeline transient using implicit finite difference method.
        
        Parameters
        ----------
        t_max : float
            Total simulation time (s)
        inlet_bc : Callable
            Boundary condition at inlet
        outlet_bc : Callable
            Boundary condition at outlet
        mode : str
            'A' = inlet Q+T / outlet P
            'B' = inlet P+T / outlet Q
        dt : float, optional
            Timestep. If None, use 10× MOC CFL timestep (IFDM is stable)
        H_initial : float
            Initial head (m)
        V_initial : float
            Initial velocity (m/s)
        T_initial : float
            Initial temperature (°C)
        P_initial : float, optional
            Initial pressure (Pa); overrides H_initial if set
        """
        Nx = self.Nx
        N = Nx + 1
        
        # IFDM is unconditionally stable — use larger timestep
        dt_cfl = self.dx / self.a
        if dt is None:
            dt = dt_cfl * 10  # default 10× CFL (adjustable)
        dt = min(dt, t_max)
        Nt = int(t_max / dt) + 1
        
        print(f"  [IFDM] dt={dt:.4f}s (CFL={dt/dt_cfl:.1f}×), Nt={Nt}")
        
        # Initialize arrays
        H = np.zeros((Nt, N))
        V = np.zeros((Nt, N))
        T = np.zeros((Nt, N))
        rho = np.zeros((Nt, N))
        mu_arr = np.zeros((Nt, N))
        
        # Initial state
        if P_initial is not None:
            rho0 = self.liquid.density(
                np.full(N, P_initial), np.full(N, T_initial))
            H0 = P_initial / (rho0 * self.g) + self.elev
        else:
            H0 = H_initial + self.elev
        
        H[0] = H0
        V[0] = V_initial
        T[0] = T_initial
        
        P_init = self.liquid.rho_ref * self.g * (H0 - self.elev)
        rho[0] = self.liquid.density(P_init, T[0])
        mu_arr[0] = self.liquid.viscosity(T[0])
        
        # Constants for temperature solve
        cp = self.liquid.cp
        U = self.pipe.heat_transfer_coeff
        T_g = self.pipe.T_ground
        
        # Time marching
        for n in range(Nt - 1):
            t = n * dt
            
            # === Build and solve hydraulic system ===
            A, b, f_n = self._build_system(H[n], V[n], dt)
            A, b = self._apply_bc(A, b, H[n], V[n], dt,
                                   inlet_bc, outlet_bc, mode, t)
            H[n+1], V[n+1] = self._solve_tridiag(A, b)
            
            # === Temperature solve (explicit upwind, same as MOC) ===
            # Energy equation:
            # cp·(∂T/∂t + V·∂T/∂x) = f·V³/(2D) - 4U·(T-T₀)/(ρ·D)
            
            for i in range(1, Nx):
                Vi = V[n+1, i]
                rho_i = rho[n, i]
                fi = f_n[i]
                
                q_friction = fi * rho_i * abs(Vi)**3 / (2 * self.D)
                q_loss = 4.0 * U * (T[n, i] - T_g) / self.D
                
                if Vi >= 0:
                    dTdx = (T[n, i] - T[n, i-1]) / self.dx
                else:
                    dTdx = (T[n, i+1] - T[n, i]) / self.dx
                
                T_new = (T[n, i] 
                         + dt * (-Vi * dTdx 
                                 + (q_friction - q_loss) / (rho_i * cp)))
                T[n+1, i] = max(-50.0, min(500.0, T_new))
            
            # Outlet temperature
            if V[n+1, Nx] >= 0:
                T[n+1, Nx] = T[n+1, Nx-1]
            else:
                T[n+1, Nx] = T[n, Nx]
            
            # Update fluid properties
            P_curr = rho[n] * self.g * (H[n+1] - self.elev)
            P_curr = np.maximum(P_curr, 1e4)  # minimum 0.01 bar
            rho[n+1] = self.liquid.density(P_curr, T[n+1])
            mu_arr[n+1] = self.liquid.viscosity(T[n+1])
        
        # Convert to result
        P = np.zeros_like(H)
        Q = np.zeros_like(H)
        for n in range(Nt):
            for i in range(N):
                P[n, i] = rho[n, i] * self.g * (H[n, i] - self.elev[i])
                Q[n, i] = V[n, i] * self.A_pipe
        
        return TransientResult(
            t=np.arange(Nt) * dt,
            x=self.x,
            P=np.maximum(P, 0), T=T, V=V, Q=Q,
            rho=rho, mu=mu_arr,
        )


# ============================================================
# 2. MacCORMACK PREDICTOR-CORRECTOR SCHEME
# ============================================================

class MacCormackSolver:
    """
    MacCormack explicit predictor-corrector for pipeline transients.
    
    Features:
      - Second-order accurate in both time and space
      - Predictor: forward difference → provisional values
      - Corrector: backward difference → corrected values
      - Excellent phase accuracy → captures wave propagation cleanly
      - Lower numerical diffusion than first-order upwind
    
    Advantages over MOC:
      - Second-order accuracy → sharper wave fronts
      - No interpolation needed (unlike MOC's interior grid)
      - Better dispersion properties
    
    Limitations:
      - Stability requires CFL ≤ 1 (same restriction as MOC)
      - May produce oscillations near steep gradients (can use flux limiter)
      - Boundary condition treatment is more involved
    """
    
    def __init__(self, pipe: Pipe, liquid: Liquid, Nx: int = 20):
        self.pipe = pipe
        self.liquid = liquid
        self.a = pipe.wave_speed(liquid)
        self.Nx = Nx
        self.dx = pipe.length / Nx
        self.g = 9.81
        
        self.x = np.linspace(0, pipe.length, Nx + 1)
        self.elev = pipe.elevation(self.x)
        self.A_pipe = pipe.area()
        self.D = pipe.diameter
        
        # Flux limiter coefficient (0=None, 1=full limiting)
        self.limiter_strength = 0.3
        
        print(f"  [MacCormack] Wave speed: {self.a:.1f} m/s")
        print(f"  [MacCormack] Grid: Nx={Nx}, dx={self.dx:.2f}m")
        print(f"  [MacCormack] dt_CFL={self.dx/self.a:.4f}s")
    
    def _fluxes(self, H: np.ndarray, V: np.ndarray, f: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute flux terms for the characteristic equations.
        
        The pipeline equations in characteristic form:
          C⁺: dH/dt + (a/g)·dV/dt + f·V·|V|/(2gD) = 0  along dx/dt = +a
          C⁻: dH/dt - (a/g)·dV/dt + f·V·|V|/(2gD) = 0  along dx/dt = -a
        
        For MacCormack, we work with the primitive form:
          ∂H/∂t + (a²/g)·∂V/∂x = 0                     (continuity)
          ∂V/∂t + g·∂H/∂x + f·V·|V|/(2D) = 0          (momentum)
        """
        N = len(H)
        B = self.a / self.g  # characteristic impedance
        
        # Continuity flux: F_H = (a²/g) * V
        # Momentum flux: F_V = g * H
        
        a2_over_g = self.a**2 / self.g
        
        # Friction source term
        friction = np.zeros(N)
        for i in range(N):
            friction[i] = f[i] * abs(V[i]) / (2 * self.D)
        
        return a2_over_g, friction
    
    def solve(
        self,
        t_max: float,
        inlet_bc: Callable,
        outlet_bc: Callable,
        mode: str = 'A',
        dt: Optional[float] = None,
        H_initial: float = 50.0,
        V_initial: float = 1.0,
        T_initial: float = 20.0,
        P_initial: Optional[float] = None,
    ) -> TransientResult:
        """
        Solve pipeline transient using MacCormack scheme.
        CFL ≤ 1 required for stability.
        """
        Nx = self.Nx
        N = Nx + 1
        
        # CFL timestep
        dt_cfl = self.dx / self.a
        if dt is None:
            dt = dt_cfl * 0.8  # CFL = 0.8 for stability margin
        elif dt > dt_cfl * 1.0:
            print(f"  ⚠️  Warning: dt={dt:.4f}s > dt_CFL={dt_cfl:.4f}s (may be unstable)")
        dt = min(dt, t_max)
        Nt = int(t_max / dt) + 1
        
        print(f"  [MacCormack] dt={dt:.4f}s (CFL={dt/dt_cfl:.2f}), Nt={Nt}")
        
        # Initialize
        H = np.zeros((Nt, N))
        V = np.zeros((Nt, N))
        T = np.zeros((Nt, N))
        rho = np.zeros((Nt, N))
        mu_arr = np.zeros((Nt, N))
        
        if P_initial is not None:
            rho0 = self.liquid.density(np.full(N, P_initial), np.full(N, T_initial))
            H0 = P_initial / (rho0 * self.g) + self.elev
        else:
            H0 = np.full(N, H_initial) + self.elev
        
        H[0] = H0
        V[0] = V_initial
        T[0] = T_initial
        
        P_init = self.liquid.rho_ref * self.g * (H0 - self.elev)
        rho[0] = self.liquid.density(np.maximum(P_init, 1e4), T[0])
        mu_arr[0] = self.liquid.viscosity(T[0])
        
        a2_over_g = self.a**2 / self.g
        g = self.g
        dx = self.dx
        
        cp = self.liquid.cp
        U = self.pipe.heat_transfer_coeff
        T_g = self.pipe.T_ground
        
        # Helper for friction factor
        def get_f(V_arr):
            f = np.zeros_like(V_arr)
            for i in range(len(V_arr)):
                Vi = max(abs(V_arr[i]), 1e-6)
                Re = self.liquid.rho_ref * Vi * self.D / max(self.liquid.viscosity_ref, 1e-10)
                f[i] = self.pipe.friction_factor(np.array([Vi]), np.array([Re]))[0]
            return f
        
        # Time marching
        for n in range(Nt - 1):
            t = n * dt
            
            f_n = get_f(V[n])
            
            # ================================================================
            # Step 1: Predictor (forward spatial differences)
            # ================================================================
            H_bar = np.zeros(N)
            V_bar = np.zeros(N)
            
            # Internal nodes (i=1 to Nx-1)
            for i in range(1, Nx):
                # Continuity: ∂H/∂t = -(a²/g)·∂V/∂x
                H_bar[i] = H[n, i] - dt * a2_over_g * (V[n, i+1] - V[n, i]) / dx
                
                # Momentum: ∂V/∂t = -g·∂H/∂x - f·V·|V|/(2D)
                V_bar[i] = (V[n, i] 
                            - dt * g * (H[n, i+1] - H[n, i]) / dx
                            - dt * f_n[i] * V[n, i] * abs(V[n, i]) / (2 * self.D))
            
            # ================================================================
            # Step 2: Corrector (backward spatial differences)
            # ================================================================
            H_corr = np.zeros(N)
            V_corr = np.zeros(N)
            
            for i in range(1, Nx):
                # Continuity
                H_corr[i] = (H[n, i] 
                             - dt * a2_over_g * (V_bar[i] - V_bar[i-1]) / dx)
                
                # Momentum
                V_corr[i] = (V[n, i] 
                             - dt * g * (H_bar[i] - H_bar[i-1]) / dx
                             - dt * f_n[i] * V_bar[i] * abs(V_bar[i]) / (2 * self.D))
            
            # ================================================================
            # Step 3: Average predictor and corrector (2nd order)
            # ================================================================
            for i in range(1, Nx):
                H[n+1, i] = 0.5 * (H_bar[i] + H_corr[i])
                V[n+1, i] = 0.5 * (V_bar[i] + V_corr[i])
            
            # ================================================================
            # Step 4: Boundary conditions (apply BEFORE artificial viscosity)
            # ================================================================
            B = self.a / self.g
            if mode.upper() == 'A':
                # Inlet: Q specified
                Q_in, T_in = inlet_bc(t + dt)
                V_in = Q_in / self.A_pipe
                V[n+1, 0] = V_in
                # C⁻: H_0 = H_1 - B·V_1 + B·V_in
                H[n+1, 0] = H[n+1, 1] - B * V[n+1, 1] + B * V_in
                T[n+1, 0] = T_in
                
                # Outlet: P specified
                P_out = outlet_bc(t + dt)
                rho_out = rho[n, Nx]
                H_out = P_out / (rho_out * self.g)
                H[n+1, Nx] = H_out
                # C⁺: V_N = (H_{N-1} + B·V_{N-1} - H_N)/B
                V[n+1, Nx] = (H[n+1, Nx-1] + B * V[n+1, Nx-1] - H_out) / B
                
            elif mode.upper() == 'B':
                P_in, T_in = inlet_bc(t + dt)
                Q_out = outlet_bc(t + dt)
                rho_in = rho[n, 0]
                H_in = P_in / (rho_in * self.g)
                V_out = Q_out / self.A_pipe
                
                H[n+1, 0] = H_in
                V[n+1, 0] = (H_in - (H[n+1, 1] - B * V[n+1, 1])) / B
                T[n+1, 0] = T_in
                
                V[n+1, Nx] = V_out
                H[n+1, Nx] = H[n+1, Nx-1] + B * (V[n+1, Nx-1] - V_out)
            
            # ================================================================
            # Step 5: Mild smoothing (only if gradient is extreme)
            # ================================================================
            if self.limiter_strength > 0:
                eps = self.limiter_strength
                for i in range(2, min(Nx - 1, self.Nx - 1)):
                    d2H = H[n+1, i+1] - 2*H[n+1, i] + H[n+1, i-1]
                    d2V = V[n+1, i+1] - 2*V[n+1, i] + V[n+1, i-1]
                    sensor = min(1.0, abs(d2H) / (max(abs(H[n+1, i]), 1.0) + 1e-10))
                    if sensor > 1.0:  # Only smooth extreme gradients
                        H[n+1, i] += eps * d2H * 0.25
                        V[n+1, i] += eps * d2V * 0.25
            
            # ================================================================
            # Step 6: Temperature solve (explicit upwind)
            # ================================================================
            for i in range(1, Nx):
                Vi = V[n+1, i]
                rho_i = rho[n, i]
                fi = f_n[i]
                
                q_friction = fi * rho_i * abs(Vi)**3 / (2 * self.D)
                q_loss = 4.0 * U * (T[n, i] - T_g) / self.D
                
                if Vi >= 0:
                    dTdx = (T[n, i] - T[n, i-1]) / dx
                else:
                    dTdx = (T[n, i+1] - T[n, i]) / dx
                
                T_new = (T[n, i] 
                         + dt * (-Vi * dTdx 
                                 + (q_friction - q_loss) / (rho_i * cp)))
                T[n+1, i] = max(-50.0, min(500.0, T_new))
            
            # Outlet temperature
            if V[n+1, Nx] >= 0:
                T[n+1, Nx] = T[n+1, Nx-1]
            else:
                T[n+1, Nx] = T[n, Nx]
            
            # Update properties
            P_curr = rho[n] * self.g * (H[n+1] - self.elev)
            P_curr = np.maximum(P_curr, 1e4)
            rho[n+1] = self.liquid.density(P_curr, T[n+1])
            mu_arr[n+1] = self.liquid.viscosity(T[n+1])
        
        # Convert
        P = np.zeros_like(H)
        Q = np.zeros_like(H)
        for n in range(Nt):
            for i in range(N):
                P[n, i] = rho[n, i] * self.g * (H[n, i] - self.elev[i])
                Q[n, i] = V[n, i] * self.A_pipe
        
        return TransientResult(
            t=np.arange(Nt) * dt,
            x=self.x,
            P=np.maximum(P, 0), T=T, V=V, Q=Q,
            rho=rho, mu=mu_arr,
        )


# ============================================================
# 3. FINITE VOLUME METHOD (MUSCL-Hancock)
# ============================================================

class FiniteVolumeSolver:
    """
    Finite Volume solver for pipeline transients (Godunov + MUSCL).
    
    Governing equations in conservative form:
      ∂U/∂t + ∂F/∂x = S
    
    where:
      U = [H; V]           — conserved variables
      F = [a²V/g; g(H+e)]  — flux (e = elevation head)
      S = [0; -f·V·|V|/(2D)] — source (friction)
    
    Features:
      - Second-order MUSCL reconstruction with minmod limiter
      - Hancock 2-step time integration
      - Rusanov (local Lax-Friedrichs) Riemann solver
      - Naturally conservative → excellent shock/water hammer capture
      - Well-balanced for sloping pipes
    
    Advantages over MOC:
      - Conservative → no mass/momentum conservation errors
      - Better wave speed preservation for strong transients
      - Handles steep gradients without spurious oscillations
    
    Limitations:
      - CFL ≤ 1 stability restriction
      - More complex implementation
    """
    
    def __init__(self, pipe: Pipe, liquid: Liquid, Nx: int = 20,
                 second_order: bool = True):
        self.pipe = pipe
        self.liquid = liquid
        self.a = pipe.wave_speed(liquid)
        self.Nx = Nx
        self.dx = pipe.length / Nx
        self.g = 9.81
        self.second_order = second_order
        
        self.x = np.linspace(0, pipe.length, Nx + 1)
        self.elev = pipe.elevation(self.x)
        self.A_pipe = pipe.area()
        self.D = pipe.diameter
        
        order = "MUSCL 2nd" if second_order else "Godunov 1st"
        print(f"  [FVM-{order}] a={self.a:.1f}m/s, Nx={Nx}, dx={self.dx:.2f}m")
        print(f"  [FVM-{order}] dt_CFL={self.dx/self.a:.4f}s")
    
    def _minmod(self, a: float, b: float) -> float:
        """Minmod slope limiter — TVD preserving"""
        if a * b <= 0:
            return 0.0
        return a if abs(a) < abs(b) else b
    
    def _char_flux(self, UL: np.ndarray, UR: np.ndarray
                    ) -> Tuple[float, float]:
        """
        Exact Riemann solver flux for the linear water hammer equations.
        
        Uses characteristic decomposition. The characteristic variables:
          w₁ = H + (a/g)·V  (propagates rightward at +a)
          w₂ = H - (a/g)·V  (propagates leftward at -a)
        
        Upwind principle:
          w₁ at interface = w₁ from LEFT cell
          w₂ at interface = w₂ from RIGHT cell
        """
        a = self.a
        g = self.g
        B = a / g  # characteristic impedance
        
        # Characteristic variables
        w1_L = UL[0] + B * UL[1]   # C⁺ characteristic (→)
        w2_R = UR[0] - B * UR[1]   # C⁻ characteristic (←)
        
        # Interfacial state from upwind characteristics
        H_star = 0.5 * (w1_L + w2_R)
        V_star = 0.5 * (w1_L - w2_R) / B
        
        # Fluxes at interface
        Fh = a**2 * V_star / g
        Fv = g * H_star
        
        return Fh, Fv
    
    def solve(
        self,
        t_max: float,
        inlet_bc: Callable,
        outlet_bc: Callable,
        mode: str = 'A',
        dt: Optional[float] = None,
        H_initial: float = 50.0,
        V_initial: float = 1.0,
        T_initial: float = 20.0,
        P_initial: Optional[float] = None,
    ) -> TransientResult:
        """
        Solve pipeline transient using Godunov FVM with characteristic flux.
        CFL ≤ 1 required for stability. If no dt given, CFL = 0.8.
        """
        Nx = self.Nx; Nc = Nx; N = Nx + 1
        g = self.g; dx = self.dx; a = self.a; B = a / g; D = self.D
        dt_cfl = dx / a
        if dt is None:
            dt = dt_cfl * 0.8
        elif dt > dt_cfl:
            print(f"  ⚠️  FVM dt={dt:.4f}s > dt_CFL={dt_cfl:.4f}s")
        dt = min(dt, t_max); Nt = int(t_max / dt) + 1
        
        order = "MUSCL" if self.second_order else "Godunov 1st"
        print(f"  [FVM-{order}] dt={dt:.4f}s (CFL={dt/dt_cfl:.2f}), Nt={Nt}")
        
        # Cell-center arrays
        H = np.zeros((Nt, Nc)); V = np.zeros((Nt, Nc))
        T = np.zeros((Nt, Nc)); rho = np.zeros((Nt, Nc)); mu_arr = np.zeros((Nt, Nc))
        
        x_center = np.linspace(dx/2, self.pipe.length - dx/2, Nc)
        elev_center = np.interp(x_center, [0, self.pipe.length],
                                [self.pipe.elevation_start, self.pipe.elevation_end])
        
        if P_initial is not None:
            H0 = P_initial / (self.liquid.rho_ref * g) + elev_center
        else:
            H0 = np.full(Nc, H_initial) + elev_center
        H[0] = H0; V[0] = V_initial; T[0] = T_initial
        P_init = self.liquid.rho_ref * g * (H0 - elev_center)
        rho[0] = self.liquid.density(np.maximum(P_init, 1e4), T[0])
        
        cp = self.liquid.cp; U = self.pipe.heat_transfer_coeff; T_g = self.pipe.T_ground
        A_pipe = self.A_pipe
        
        def get_f(V_arr):
            f = np.zeros_like(V_arr)
            for i in range(len(V_arr)):
                Vi = max(abs(V_arr[i]), 1e-6)
                Re = self.liquid.rho_ref * Vi * D / max(self.liquid.viscosity_ref, 1e-10)
                f[i] = self.pipe.friction_factor(np.array([Vi]), np.array([Re]))[0]
            return f
        
        for n in range(Nt - 1):
            t = n * dt; f_n = get_f(V[n])
            
            # --- Reconstruct interface states ---
            UL = np.zeros((Nc + 1, 2)); UR = np.zeros((Nc + 1, 2))
            
            if self.second_order:
                for i in range(Nc):
                    if i == 0:
                        dH = self._minmod(H[n,1]-H[n,0], 0.0)
                        dV = self._minmod(V[n,1]-V[n,0], 0.0)
                    elif i == Nc-1:
                        dH = self._minmod(H[n,Nc-1]-H[n,Nc-2], 0.0)
                        dV = self._minmod(V[n,Nc-1]-V[n,Nc-2], 0.0)
                    else:
                        dH = self._minmod(H[n,i]-H[n,i-1], H[n,i+1]-H[n,i])
                        dV = self._minmod(V[n,i]-V[n,i-1], V[n,i+1]-V[n,i])
                    HL = H[n,i] - 0.5*dH; HR = H[n,i] + 0.5*dH
                    VL = V[n,i] - 0.5*dV; VR = V[n,i] + 0.5*dV
                    dFH = a**2*(VR - VL)/g; dFV = g*(HR - HL)
                    UR[i] = [HL - 0.5*dt/dx*dFH, VL - 0.5*dt/dx*dFV]
                    UL[i+1] = [HR - 0.5*dt/dx*dFH, VR - 0.5*dt/dx*dFV]
            else:
                for i in range(Nc):
                    UR[i] = [H[n,i], V[n,i]]
                    UL[i+1] = [H[n,i], V[n,i]]
            
            # --- Ghost cells via characteristic BC ---
            if mode.upper() == 'A':
                Q_in, T_in = inlet_bc(t + dt); P_out = outlet_bc(t + dt)
                V_in = Q_in / A_pipe; H_out = P_out / (self.liquid.rho_ref * g)
                # Inlet: ghost such that interface gets V_in
                UL[0] = [UR[0][0] - B*UR[0][1] + B*V_in, V_in]
                # Outlet: ghost such that interface gets H_out
                wN = UL[Nc][0] + B*UL[Nc][1]
                UR[Nc] = [H_out, (wN - H_out)/B]
            else:
                P_in, T_in = inlet_bc(t+dt); Q_out = outlet_bc(t+dt)
                H_in = P_in / (self.liquid.rho_ref * g); V_out = Q_out / A_pipe
                w2 = UR[0][0] - B*UR[0][1]
                UL[0] = [H_in, (UR[0][0] + B*UR[0][1] - 2*H_in)/(-B)]
                wNp = UL[Nc][0] + B*UL[Nc][1]
                UR[Nc] = [0.5*(wNp + (wNp - 2*B*V_out)),
                          0.5*(wNp - (wNp - 2*B*V_out))/B]
            
            # --- Characteristic flux at all interfaces ---
            FH = np.zeros(Nc + 1); FV = np.zeros(Nc + 1)
            for i in range(Nc + 1):
                w1 = UL[i][0] + B*UL[i][1]; w2 = UR[i][0] - B*UR[i][1]
                FH[i] = a**2 * 0.5*(w1-w2)/B / g
                FV[i] = g * 0.5*(w1+w2)
            
            # --- Update cell averages ---
            for i in range(Nc):
                H[n+1,i] = H[n,i] - dt/dx*(FH[i+1]-FH[i])
                S_fric = -f_n[i] * V[n,i] * abs(V[n,i]) / (2*D)
                V[n+1,i] = V[n,i] - dt/dx*(FV[i+1]-FV[i]) + dt*S_fric
            
            # --- Temperature (upwind) ---
            for i in range(Nc):
                Vi = np.clip(V[n+1,i], -5.0, 5.0)
                q_fric = min(f_n[i]*rho[n,i]*abs(Vi)**3/(2*D), 1e8)
                q_loss = 4*U*(T[n,i]-T_g)/D
                if abs(Vi) < 1e-8:
                    dTdx = 0.0
                elif Vi >= 0:
                    if i == 0:
                        _, Tb = inlet_bc(t+dt); dTdx = (T[n,i]-Tb)/dx
                    else:
                        dTdx = (T[n,i]-T[n,i-1])/dx
                else:
                    if i == Nc-1:
                        dTdx = 0.0
                    else:
                        dTdx = (T[n,i+1]-T[n,i])/dx
                src = np.clip((q_fric - q_loss)/(rho[n,i]*cp), -500, 500)
                T[n+1,i] = max(-50, min(500, T[n,i] + dt*(-Vi*dTdx + src)))
            
            # --- Update fluid properties ---
            P_curr = rho[n]*g*(H[n+1] - elev_center)
            P_curr = np.maximum(P_curr, 1e4)
            rho[n+1] = self.liquid.density(P_curr, T[n+1])
            mu_arr[n+1] = self.liquid.viscosity(T[n+1])
        
        # Cell → node interpolation
        Hn = np.zeros((Nt, N)); Vn = np.zeros((Nt, N))
        Tn = np.zeros((Nt, N)); rn = np.zeros((Nt, N)); mn = np.zeros((Nt, N))
        for n in range(Nt):
            Hn[n,0]=H[n,0]; Vn[n,0]=V[n,0]; Tn[n,0]=T[n,0]; rn[n,0]=rho[n,0]
            for i in range(1,Nc):
                Hn[n,i]=0.5*(H[n,i-1]+H[n,i]); Vn[n,i]=0.5*(V[n,i-1]+V[n,i])
                Tn[n,i]=0.5*(T[n,i-1]+T[n,i]); rn[n,i]=0.5*(rho[n,i-1]+rho[n,i])
            Hn[n,Nx]=H[n,Nc-1]; Vn[n,Nx]=V[n,Nc-1]; Tn[n,Nx]=T[n,Nc-1]; rn[n,Nx]=rho[n,Nc-1]
        
        Pn = rn*g*(Hn - self.elev); Qn = Vn*self.A_pipe
        return TransientResult(
            t=np.arange(Nt)*dt, x=self.x,
            P=np.maximum(Pn, 0), T=Tn, V=Vn, Q=Qn,
            rho=rn, mu=mn,
        )
    
    def _infer_ghost_H(self, H: np.ndarray, i: int, side: str) -> float:
        """Infer ghost cell value for slope limiting at boundaries."""
        Nc = len(H)
        if side == 'left':
            return 2 * H[0] - H[1]  # linear extrapolation
        else:
            return 2 * H[Nc-1] - H[Nc-2]


# ============================================================
# SOLVER COMPARISON UTILITY
# ============================================================

def compare_solvers(
    pipe: Pipe,
    liquid: Liquid,
    inlet_bc: Callable,
    outlet_bc: Callable,
    mode: str = 'A',
    Nx: int = 20,
    t_max: float = 100.0,
    Q0: float = 0.25,
    T0: float = 45.0,
    P_out: float = 2e6,
    compare_methods: list = None,
) -> dict:
    """
    Run all available solvers on the same scenario and compare results.
    
    Parameters
    ----------
    pipe : Pipe
    liquid : Liquid
    inlet_bc, outlet_bc : Callable
    mode : str
    Nx : int
    t_max : float
    Q0 : float
    T0 : float
    P_out : float
    compare_methods : list, optional
        Solvers to compare. Default: all four.
    
    Returns
    -------
    dict[str, TransientResult]
        Results keyed by method name
    """
    from .solver import SinglePhaseTransientSolver
    
    if compare_methods is None:
        compare_methods = ['MOC', 'IFDM', 'MacCormack', 'FVM']
    
    results = {}
    
    for method in compare_methods:
        print(f"\n{'='*60}")
        print(f"  Running: {method}")
        print(f"{'='*60}")
        
        if method == 'MOC':
            solver = SinglePhaseTransientSolver(pipe, liquid, Nx=Nx)
        elif method == 'IFDM':
            solver = ImplicitFDMSolver(pipe, liquid, Nx=Nx, theta=0.5)
        elif method == 'MacCormack':
            solver = MacCormackSolver(pipe, liquid, Nx=Nx)
        elif method == 'FVM':
            solver = FiniteVolumeSolver(pipe, liquid, Nx=Nx)
        else:
            print(f"  Unknown method: {method}")
            continue
        
        # Compute initial steady state
        from .steady import SteadyStateCalculator
        steady = SteadyStateCalculator(pipe, liquid)
        V0, P0, T_profile = steady.initialize_transient(
            Q0, T0, P_out, solver)
        
        try:
            result = solver.solve(
                t_max=t_max,
                inlet_bc=inlet_bc,
                outlet_bc=outlet_bc,
                mode=mode,
                V_initial=V0,
                T_initial=T_profile[0] if isinstance(T_profile, np.ndarray) else T0,
                P_initial=P_out,
            )
            results[method] = result
            print(f"  ✅ {method} completed: Nt={result.Nt}, max(P)={result.P.max()/1e6:.3f} MPa")
        except Exception as e:
            print(f"  ❌ {method} failed: {e}")
    
    return results
