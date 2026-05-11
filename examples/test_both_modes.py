"""
test_both_modes.py — Test both boundary condition modes
Mode A: Inlet Q+T / Outlet P
Mode B: Inlet P+T / Outlet Q
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sim.fluid import Liquid
from sim.pipe import Pipe
from sim.solver import (
    SinglePhaseTransientSolver,
    flow_inlet, pressure_inlet,
    pressure_outlet, flow_outlet,
    step
)
import json

# ============================================================
# Common setup
# ============================================================
pipe = Pipe(
    length=5000.0,
    diameter=0.4,
    wall_thickness=0.01,
    roughness=4.5e-5,
    heat_transfer_coeff=8.0,
    T_ground=8.0,
    elevation_start=0.0,
    elevation_end=20.0,
)

liquid = Liquid(name="Crude Oil",
    rho_ref=850.0,
    bulk_modulus=1.8e9,
    cp=2000.0,
    viscosity_ref=8e-3,
    thermal_expansion=8.0e-4,
    T_ref=20.0)

print(f"{'='*60}")
print(f"Crude Oil Pipeline Transient Simulation")
print(f"{'='*60}")
print(f"Pipe: L={pipe.length}m, D={pipe.diameter}m, Δz={pipe.elevation_end}m")
print(f"Fluid: {liquid.name}, ρ={liquid.rho_ref}kg/m³, μ={liquid.viscosity_ref}Pa·s")
print()

# ============================================================
# Test 1: Mode A — Inlet flow rate + temperature  |  Outlet pressure
# ============================================================
print(f"{'─'*60}")
print(f"Test 1: Mode A — Inlet Q+T / Outlet P")
print(f"{'─'*60}")

Q0 = 0.15  # m³/s
P_out = 1.0e6  # 1 MPa at outlet

# Inlet: step change in flow rate from 0.15 → 0.10 m³/s at t=10s
Q_in_func = lambda t: step(Q0, Q0 * 0.67, 5.0, 2.0)(t)
T_in_func = lambda t: 30.0  # constant 30°C inlet

# Outlet: constant pressure
P_out_func = lambda t: P_out

solver_a = SinglePhaseTransientSolver(pipe, liquid, Nx=25)
result_a = solver_a.solve(
    t_max=30.0,
    inlet_bc=flow_inlet(Q_in_func, T_in_func),
    outlet_bc=pressure_outlet(P_out_func),
    mode='A',
    V_initial=Q0 / pipe.area(),
    T_initial=25.0,
    P_initial=P_out + 2e5,  # slightly higher at inlet
)

# Extract key metrics
peak_P = np.max(result_a.P[10:, :]) / 1e6
steady_P_inlet = result_a.P[-1, 0] / 1e6
steady_T_outlet = result_a.T[-1, -1]

print(f"  Initial flow: {Q0:.3f} m³/s → reduced to {Q0*0.67:.3f} m³/s")
print(f"  Peak pressure: {peak_P:.2f} MPa")
print(f"  Final inlet pressure: {steady_P_inlet:.2f} MPa")
print(f"  Final outlet temperature: {steady_T_outlet:.1f} °C")
print(f"  Inlet temp: 30°C, Ground temp: {pipe.T_ground}°C → temp drop: {30 - steady_T_outlet:.1f}°C")
print(f"  ✅ Mode A test complete")
print()

# ============================================================
# Test 2: Mode B — Inlet pressure + temperature  |  Outlet flow rate
# ============================================================
print(f"{'─'*60}")
print(f"Test 2: Mode B — Inlet P+T / Outlet Q")
print(f"{'─'*60}")

P_in = 1.5e6     # 1.5 MPa at inlet
Q_target = 0.12  # m³/s

# Inlet: pressure drops from 1.5 → 1.2 MPa at t=8s
P_in_func = lambda t: step(P_in, P_in * 0.8, 8.0, 3.0)(t)
T_in_func = lambda t: 35.0

# Outlet: step change in flow from 0.12 → 0.06 m³/s
Q_out_func = lambda t: step(Q_target, Q_target * 0.5, 5.0, 1.0)(t)

solver_b = SinglePhaseTransientSolver(pipe, liquid, Nx=25)
result_b = solver_b.solve(
    t_max=25.0,
    inlet_bc=pressure_inlet(P_in_func, T_in_func),
    outlet_bc=flow_outlet(Q_out_func),
    mode='B',
    V_initial=Q_target / pipe.area(),
    T_initial=28.0,
    P_initial=P_in,
)

peak_P_b = np.max(result_b.P[:, :]) / 1e6
final_P_out = result_b.P[-1, -1] / 1e6
temp_drop = 35.0 - result_b.T[-1, -1]

print(f"  Inlet pressure: {P_in/1e6:.1f} → {P_in*0.8/1e6:.1f} MPa")
print(f"  Outlet flow: {Q_target:.3f} → {Q_target*0.5:.3f} m³/s")
print(f"  Peak pressure: {peak_P_b:.2f} MPa")
print(f"  Final outlet pressure: {final_P_out:.2f} MPa")
print(f"  Temperature drop: {temp_drop:.1f}°C")
print(f"  ✅ Mode B test complete")
print()

# ============================================================
# Save summary
# ============================================================
summary = {
    "pipe": {
        "length_m": pipe.length,
        "diameter_m": pipe.diameter,
        "elevation_change_m": pipe.elevation_end - pipe.elevation_start,
    },
    "fluid": {"name": liquid.name, "density": liquid.rho_ref},
    "wave_speed_mps": solver_a.a,
    "tests": {
        "mode_A": {
            "description": "Inlet Q+T / Outlet P",
            "peak_pressure_mpa": round(peak_P, 3),
            "final_inlet_pressure_mpa": round(steady_P_inlet, 3),
            "temp_drop_c": round(30 - steady_T_outlet, 2),
        },
        "mode_B": {
            "description": "Inlet P+T / Outlet Q",
            "peak_pressure_mpa": round(peak_P_b, 3),
            "final_outlet_pressure_mpa": round(final_P_out, 3),
            "temp_drop_c": round(temp_drop, 2),
        }
    }
}

with open(os.path.join(os.path.dirname(__file__), '..', 'output', 'test_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"{'='*60}")
print(f"✅ Both tests complete! Results saved.")
print(f"{'='*60}")

# ============================================================
# Plot
# ============================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# Column 1: Mode A
ax = axes[0, 0]
ax.plot(result_a.t, result_a.Q[:, 0] * 1000, 'b-', label='Inlet')
ax.plot(result_a.t, result_a.Q[:, -1] * 1000, 'r--', label='Outlet')
ax.set_ylabel('Flow rate (L/s)')
ax.set_title('Mode A: Flow rate (Inlet specified)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(result_a.t, result_a.P[:, 0] / 1e6, 'b-', label='Inlet')
ax.plot(result_a.t, result_a.P[:, -1] / 1e6, 'r--', label='Outlet')
ax.set_ylabel('Pressure (MPa)')
ax.set_title('Mode A: Pressure (Outlet specified)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[2, 0]
ax.plot(result_a.t, result_a.T[:, 0], 'b-', label='Inlet')
ax.plot(result_a.t, result_a.T[:, -1], 'r--', label='Outlet')
ax.set_ylabel('Temperature (°C)')
ax.set_xlabel('Time (s)')
ax.set_title('Mode A: Temperature')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Column 2: Mode B
ax = axes[0, 1]
ax.plot(result_b.t, result_b.Q[:, 0] * 1000, 'b-', label='Inlet')
ax.plot(result_b.t, result_b.Q[:, -1] * 1000, 'r--', label='Outlet')
ax.set_ylabel('Flow rate (L/s)')
ax.set_title('Mode B: Flow rate (Outlet specified)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.plot(result_b.t, result_b.P[:, 0] / 1e6, 'b-', label='Inlet')
ax.plot(result_b.t, result_b.P[:, -1] / 1e6, 'r--', label='Outlet')
ax.set_ylabel('Pressure (MPa)')
ax.set_title('Mode B: Pressure (Inlet specified)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[2, 1]
ax.plot(result_b.t, result_b.T[:, 0], 'b-', label='Inlet')
ax.plot(result_b.t, result_b.T[:, -1], 'r--', label='Outlet')
ax.set_ylabel('Temperature (°C)')
ax.set_xlabel('Time (s)')
ax.set_title('Mode B: Temperature')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.suptitle('Pipeline Transient Simulation — Coupled Hydraulic-Thermal', fontsize=13)
plt.tight_layout()
plt.savefig('/root/.openclaw/workspace/projects/pipeline-sim/output/both_modes.png', dpi=150)
print("✅ Plot saved to output/both_modes.png")
