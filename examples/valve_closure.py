"""
valve_closure.py — Valve closure water hammer example
Single pipe with reservoir upstream and valve downstream
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sim.pipeline_sim import (
    Pipe, Fluid, MOCSolver,
    constant_head, valve_closure
)
import numpy as np

# ============================================================
# Case 1: Simple valve closure (same as test)
# ============================================================
pipe = Pipe(
    length=2000.0,
    diameter=0.3,
    wall_thickness=0.008,
)
fluid = Fluid()

solver = MOCSolver(pipe, fluid, Nx=40)
print(f"{'='*60}")
print(f"Pipeline Simulation: Valve Closure Water Hammer")
print(f"{'='*60}")
print(f"Pipe: L={pipe.length}m, D={pipe.diameter}m, e={pipe.wall_thickness}m")
print(f"Wave speed: {solver.a:.1f} m/s")
print(f"Grid: Nx={solver.Nx}, dx={solver.dx:.1f}m, dt={solver.dt:.4f}s")

V0 = 1.5
H0 = 100.0

H_up = constant_head(H0)
V_down = valve_closure(V0, t_close=1.0, profile='linear')

sol = solver.solve(
    t_max=8.0,
    H_upstream=H_up,
    V_downstream=V_down,
    H_initial=H0,
    V_initial=V0,
)

peak_H = np.max(sol.H)
dH_jouk = solver.a * V0 / 9.81

print(f"\n{'─'*60}")
print(f"Steady flow velocity: {V0} m/s")
print(f"Joukowsky ΔH: {dH_jouk:.1f} m")
print(f"Theoretical peak: {H0 + dH_jouk:.1f} m")
print(f"Simulated peak: {peak_H:.1f} m")
print(f"Peak pressure: {peak_H * fluid.density * 9.81 / 1e6:.2f} MPa")
print(f"{'─'*60}")

# Save results
import json
results = {
    "description": "Valve closure water hammer",
    "pipe": {"length": pipe.length, "diameter": pipe.diameter},
    "fluid": {"density": fluid.density},
    "wave_speed": solver.a,
    "peak_head": float(peak_H),
    "joukowsky_delta": float(dH_jouk),
    "peak_pressure_mpa": float(peak_H * fluid.density * 9.81 / 1e6),
}
with open('/root/.openclaw/workspace/projects/pipeline-sim/output/valve_closure_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✅ Results saved to output/")

# Plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(12, 8))

# 1. Head at valve
t = sol.t
axes[0].plot(t, sol.head_at(-1), 'b-', linewidth=1.5)
axes[0].axhline(H0, color='gray', linestyle='--', alpha=0.5, label=f'H₀={H0}m')
axes[0].axhline(H0 + dH_jouk, color='r', linestyle=':', alpha=0.5, label=f'Joukowsky {H0+dH_jouk:.0f}m')
axes[0].set_ylabel('Head at valve (m)')
axes[0].set_title('Water Hammer — Valve Closure')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# 2. Velocity at valve
axes[1].plot(t, sol.vel_at(-1), 'g-', linewidth=1.5)
axes[1].set_ylabel('Velocity at valve (m/s)')
axes[1].set_xlabel('Time (s)')
axes[1].grid(True, alpha=0.3)

# 3. Head profiles
x = np.linspace(0, pipe.length, solver.Nx + 1)
snapshots = [t_idx for t_idx in range(0, sol.Nt, int(sol.Nt / 6))]
for t_idx in snapshots[:8]:
    axes[2].plot(x, sol.H[t_idx, :], label=f't={sol.t[t_idx]:.2f}s')
axes[2].set_ylabel('Head (m)')
axes[2].set_xlabel('Distance (m)')
axes[2].set_title('Head profile along pipeline')
axes[2].legend(fontsize=7)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/root/.openclaw/workspace/projects/pipeline-sim/output/valve_closure.png', dpi=150)
print("✅ Plot saved to output/valve_closure.png")
