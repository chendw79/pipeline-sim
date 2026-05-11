#!/usr/bin/env python3
"""
Solver Comparison — Compare all 5 transient solvers on the same scenario.

Tests: valve closure (10s) in a 15km crude oil pipeline.

Methods:
  1. MOC          — Method of Characteristics (default)
  2. IFDM (CN)    — Implicit Finite Difference, Crank-Nicolson
  3. MacCormack   — Predictor-Corrector, 2nd order
  4. FVM Godunov  — Finite Volume, 1st order Godunov
  5. FVM MUSCL    — Finite Volume, 2nd order MUSCL-Hancock

Usage:
  python examples/solver_comparison.py
"""

import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim import *
from sim.solver import flow_inlet, pressure_outlet, step, SinglePhaseTransientSolver
from sim.solver_advanced import (ImplicitFDMSolver, MacCormackSolver,
                                 FiniteVolumeSolver, compare_solvers)

# ============================================================
# Pipeline config (15km crude oil pipeline)
# ============================================================
pipe = Pipe(
    length=15000, diameter=0.6, wall_thickness=0.014,
    roughness=4.5e-5, elevation_start=0.0, elevation_end=50.0,
)

liquid = Liquid(
    name="Crude Oil",
    rho_ref=860.0,          # kg/m³
    bulk_modulus=1.8e9,     # Pa
    cp=2000.0,              # J/(kg·K)
    viscosity_ref=12e-3,    # Pa·s
    thermal_expansion=8.0e-4,
)

steady = SteadyStateCalculator(pipe, liquid)

# Operating conditions
Q0 = 0.25           # m³/s (250 L/s)
T0 = 45.0           # °C inlet temperature
P_out = 2e6         # Pa outlet pressure
A = pipe.area()
V0 = Q0 / A

# ============================================================
# Scenario: Instant valve closure at t=10s
# ============================================================
print("=" * 70)
print(" PipelineSim Solver Comparison: Valve Closure (10s)")
print("=" * 70)
print(f"  Pipe: L={pipe.length/1000:.1f}km, D={pipe.diameter*1000:.0f}mm")
print(f"  Fluid: {liquid.name} (ρ={liquid.rho_ref} kg/m³, K={liquid.bulk_modulus/1e9:.1f} GPa)")
print(f"  Flow: Q={Q0*1000:.0f} L/s, V={V0:.3f} m/s")
print(f"  Wave speed: {pipe.wave_speed(liquid):.1f} m/s")
print(f"  Joukowsky peak: {P_out/1e6 + liquid.rho_ref*pipe.wave_speed(liquid)*V0/1e6:.3f} MPa")
print()

# Boundary conditions
Q_func = step(Q0, 0.0, 10.0)            # flow steps from 250→0 L/s at 10s
T_func = step(T0, T0, 0.0)              # constant inlet temperature
P_func = step(P_out, P_out, 0.0)        # constant outlet pressure

inlet_bc = flow_inlet(Q_func, T_func)
outlet_bc = pressure_outlet(P_func)

# ============================================================
# Run all solvers
# ============================================================
results = {}
solver_configs = [
    ("MOC",          SinglePhaseTransientSolver, {}),
    ("IFDM (CN)",    ImplicitFDMSolver, {"theta": 0.5}),
    ("IFDM (BE)",    ImplicitFDMSolver, {"theta": 1.0}),
    ("MacCormack",   MacCormackSolver, {}),
    ("FVM Godunov",  FiniteVolumeSolver, {"second_order": False}),
    ("FVM MUSCL",    FiniteVolumeSolver, {"second_order": True}),
]

for name, Cls, kwargs in solver_configs:
    print(f"\n--- {name} ---")
    try:
        solver = Cls(pipe, liquid, Nx=20, **kwargs)
        r = solver.solve(
            t_max=60.0,
            inlet_bc=inlet_bc,
            outlet_bc=outlet_bc,
            mode="A",
            V_initial=V0,
            T_initial=T0,
            P_initial=P_out,
        )
        results[name] = r
        print(f"  ✅ Nt={r.Nt:4d}  P=[{r.P.min()/1e6:.3f}, {r.P.max()/1e6:.3f}] MPa")
    except Exception as e:
        print(f"  ❌ {e}")
        import traceback; traceback.print_exc()

# ============================================================
# Summary table
# ============================================================
print("\n" + "=" * 70)
print(" COMPARISON SUMMARY")
print("=" * 70)
print(f" {'Method':<15s} {'Nt':>5s} {'P_min':>8s} {'P_max':>8s} {'Δt':>8s} {'Error':>8s}")
print("-" * 70)

J_pk = P_out / 1e6 + liquid.rho_ref * pipe.wave_speed(liquid) * V0 / 1e6

for name in [n for n, _, _ in solver_configs]:
    if name in results:
        r = results[name]
        dt_avg = r.t[-1] / (r.Nt - 1) if r.Nt > 1 else 0
        p_max = r.P.max() / 1e6
        error = abs(p_max - J_pk) / J_pk * 100
        print(f" {name:<15s} {r.Nt:>5d} {r.P.min()/1e6:>8.3f} {p_max:>8.3f} {dt_avg:>8.4f} {error:>7.2f}%")

print("-" * 70)
print(f" {'Joukowsky':<15s} {'-':>5s} {'-':>8s} {J_pk:>8.3f} {'-':>8s} {'0.00%':>7s}")
print()

# ============================================================
# Key takeaways
# ============================================================
print("=" * 70)
print(" KEY TAKEAWAYS")
print("=" * 70)
print("""
 • MOC:         Standard benchmark. Good accuracy. CFL=1 limits timestep.
 • IFDM (CN):   Unconditionally stable! CFL=10+ with no loss of accuracy.
                Ideal for slow transients / long-duration simulations.
 • IFDM (BE):   More diffusive than CN but unconditionally stable.
 • MacCormack:  2nd order accurate but Gibbs oscillations near shocks.
                P_max overestimates by >100% for sudden valve closure.
 • FVM Godunov: Most accurate peak pressure. Conservative scheme.
                Excellent for water hammer with strong shocks.
 • FVM MUSCL:   Slightly sharper than Godunov. Good all-around.
""")

# ============================================================
# Save results for visualization
# ============================================================
out_dir = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(out_dir, exist_ok=True)

np.savez(
    os.path.join(out_dir, "solver_comparison.npz"),
    **{name: {"t": r.t, "x": r.x, "P": r.P, "V": r.V, "T": r.T}
       for name, r in results.items()},
)

print(f"Results saved to {out_dir}/solver_comparison.npz")
print(f"Visualize with: python -c 'import numpy as np; d=np.load(…)'")
