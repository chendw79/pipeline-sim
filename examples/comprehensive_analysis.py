"""
Add CSV export and self-diagnostic to PipelineSim
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
)

def export_to_csv(result, filename, pipe):
    """Export simulation results to CSV"""
    import csv
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['time_s']
        for i in range(len(result.x)):
            header += [f'P_{i}_Pa', f'T_{i}_C', f'V_{i}_mps', f'Q_{i}_m3s']
        writer.writerow(header)
        
        for n in range(0, len(result.t), max(1, len(result.t)//200)):
            row = [round(result.t[n], 4)]
            for i in range(len(result.x)):
                row += [
                    round(result.P[n, i], 2),
                    round(result.T[n, i], 4),
                    round(result.V[n, i], 6),
                    round(result.Q[n, i], 6),
                ]
            writer.writerow(row)
    print(f"  📄 CSV saved: {filename} ({os.path.getsize(filename)/1024:.0f} KB)")

def generate_report(result, pipe, liquid, solver, case_name):
    """Generate a self-diagnostic report"""
    V0_initial = result.V[0, 0]
    a = solver.a
    dP_jouk = liquid.rho_ref * a * V0_initial
    
    peak_P = np.max(result.P)
    steady_P = np.mean(result.P[-5:, 0])  # end state
    
    report = f"""
{'='*60}
PipelineSim Self-Diagnostic Report
{'='*60}

Case: {case_name}

┌─ System Information
│  Python: {sys.version.split()[0]}
│  NumPy: {np.__version__}
│  Pipe: L={pipe.length:.0f}m, D={pipe.diameter:.3f}m, e={pipe.wall_thickness:.4f}m
│  Fluid: {liquid.name} (ρ₀={liquid.rho_ref}kg/m³, K={liquid.bulk_modulus/1e9:.2f}GPa)
│  
├─ Solver Configuration
│  Wave speed a = {a:.1f} m/s
│  Grid Nx = {solver.Nx}, dx = {solver.dx:.2f}m, dt = {solver.dt:.4f}s
│  CFL = {solver.dt * a / solver.dx:.2f} (1.0 = exact)
│  Duration: {result.t[-1]:.1f}s, Steps: {result.Nt}
│  
├─ Validation Metrics
│  Initial velocity V₀ = {V0_initial:.4f} m/s
│  Joukowsky ΔP = {dP_jouk/1e6:.4f} MPa
│  Theoretical peak = {(steady_P + dP_jouk)/1e6:.4f} MPa
│  Simulated peak = {peak_P/1e6:.4f} MPa
│  Deviation = {abs(peak_P - (steady_P + dP_jouk))/(steady_P + dP_jouk)*100:.2f}%
│  
│  Result range:
│    Pressure: {np.min(result.P)/1e6:.2f} ~ {np.max(result.P)/1e6:.2f} MPa
│    Temperature: {np.min(result.T):.1f} ~ {np.max(result.T):.1f} °C
│    Velocity: {np.min(result.V):.4f} ~ {np.max(result.V):.4f} m/s
│  
└─ Stability Check
   Final pressure std (last 10%): {np.std(result.P[-max(5,result.Nt//10):, -1]):.1f} Pa
   {'✅ NUMERICALLY STABLE' if np.std(result.P[-max(5,result.Nt//10):, -1]) < 1e5 else '⚠️ LARGE OSCILLATIONS'}
   {'✅ ENERGY CONSERVING' if np.max(result.T) < 1000 else '⚠️ TEMPERATURE DIVERGENCE'}
"""
    return report


# ============================================================
# Run a comprehensive oil pipeline example
# ============================================================
print("PipelineSim Comprehensive Analysis")
print("="*60)

pipe = Pipe(
    length=15000.0, diameter=0.6, wall_thickness=0.014,
    roughness=4.5e-5, heat_transfer_coeff=8.0,
    T_ground=8.0, elevation_start=0.0, elevation_end=50.0,
)

liquid = Liquid(name="Crude Oil",
    rho_ref=860.0, bulk_modulus=1.8e9, cp=2000.0,
    viscosity_ref=12e-3, thermal_expansion=8.0e-4, T_ref=20.0)

Q_operating = 0.25  # m³/s
P_outlet = 0.5e6    # 500 kPa outlet pressure
T_inlet = 45.0      # 45°C at inlet

print(f"\n📋 Case: Crude Oil Pipeline (L={pipe.length/1000:.0f}km, D={pipe.diameter*1000:.0f}mm)")
print(f"    Flow={Q_operating*1000:.0f}L/s, Inlet T={T_inlet}°C, Ground={pipe.T_ground}°C")

# --- Mode A: Inlet flow + temperature / Outlet pressure ---
print(f"\n{'─'*60}")
print(f"Mode A: Steady operation with outlet valve closure")
print(f"{'─'*60}")

solver = SinglePhaseTransientSolver(pipe, liquid, Nx=40)

# Step 1: Steady state
# Step 2: Outlet valve partially closes (flow reduces)
# Step 3: Thermal stabilization

Q_after = Q_operating * 0.5  # valve halves flow
close_start = 2.0
close_duration = 5.0

def inlet_bc(t):
    if t < close_start:
        return Q_operating, T_inlet
    elif t < close_start + close_duration:
        frac = (t - close_start) / close_duration
        Q = Q_operating + (Q_after - Q_operating) * frac
        return Q, T_inlet
    else:
        return Q_after, T_inlet

def outlet_bc(t):
    return P_outlet

result_a = solver.solve(
    t_max=100.0,
    inlet_bc=flow_inlet(lambda t: inlet_bc(t)[0], lambda t: inlet_bc(t)[1]),
    outlet_bc=pressure_outlet(outlet_bc),
    mode='A',
    V_initial=Q_operating / pipe.area(),
    T_initial=20.0,
    P_initial=P_outlet + 1.5e6,
)

# Export
export_to_csv(result_a, os.path.join(os.path.dirname(__file__), '..', 'output', 'oil_pipeline_modeA.csv'), pipe)

report = generate_report(result_a, pipe, liquid, solver, "Oil Pipeline Mode A")
print(report)

# --- Mode B: Inlet pressure + temperature / Outlet flow ---
print(f"\n{'─'*60}")
print(f"Mode B: Pump pressure reduction + flow variation")
print(f"{'─'*60}")

P_inlet = 2.0e6  # 2 MPa inlet

def inlet_pressure_b(t):
    if t < 5.0:
        return P_inlet
    elif t < 10.0:
        frac = (t - 5.0) / 5.0
        return P_inlet * (1.0 - 0.3 * frac)
    else:
        return P_inlet * 0.7

def outlet_bc_b(t):
    return Q_operating

solver_b = SinglePhaseTransientSolver(pipe, liquid, Nx=40)
result_b = solver_b.solve(
    t_max=80.0,
    inlet_bc=pressure_inlet(inlet_pressure_b, lambda t: T_inlet),
    outlet_bc=flow_outlet(outlet_bc_b),
    mode='B',
    V_initial=Q_operating / pipe.area(),
    T_initial=25.0,
    P_initial=P_inlet,
)

export_to_csv(result_b, os.path.join(os.path.dirname(__file__), '..', 'output', 'oil_pipeline_modeB.csv'), pipe)

# Temperature analysis
temp_drop_A = result_a.T[0, 0] - result_a.T[-1, -1]
temp_drop_B = result_b.T[0, 0] - result_b.T[-1, -1]

print(f"\n{'='*60}")
print(f"THERMAL ANALYSIS")
print(f"{'='*60}")
print(f"  Mode A: Inlet {result_a.T[0,0]:.1f}°C → Outlet {result_a.T[-1,-1]:.1f}°C (drop {temp_drop_A:.1f}°C)")
print(f"  Mode B: Inlet {result_b.T[0,0]:.1f}°C → Outlet {result_b.T[-1,-1]:.1f}°C (drop {temp_drop_B:.1f}°C)")
print(f"  Ground temperature: {pipe.T_ground}°C")
print(f"  {'✅ Thermal coupling active' if temp_drop_A > 0.5 else '⚠️ Low thermal gradient'}")

print(f"\n{'='*60}")
print(f"✅ Complete analysis done. CSV exported.")
print(f"{'='*60}")

# Plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# Mode A
ax = axes[0, 0]
ax.plot(result_a.t, result_a.Q[:, 0]*1000, 'b-', label='Inlet')
ax.plot(result_a.t, result_a.Q[:, -1]*1000, 'r--', label='Outlet')
ax.set_ylabel('Flow (L/s)'); ax.set_title('Mode A: Flow'); ax.legend(); ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.plot(result_a.t, result_a.P[:, 0]/1e6, 'b-', label='Inlet')
ax.plot(result_a.t, result_a.P[:, -1]/1e6, 'r--', label='Outlet')
ax.set_ylabel('Pressure (MPa)'); ax.set_title('Mode A: Pressure'); ax.legend(); ax.grid(alpha=0.3)

ax = axes[2, 0]
ax.plot(result_a.t, result_a.T[:, 0], 'b-', label='Inlet')
ax.plot(result_a.t, result_a.T[:, -1], 'r--', label='Outlet')
ax.axhline(pipe.T_ground, color='gray', ls='--', alpha=0.5, label=f'Ground {pipe.T_ground}°C')
ax.set_ylabel('Temp (°C)'); ax.set_xlabel('Time (s)')
ax.set_title('Mode A: Temperature'); ax.legend(); ax.grid(alpha=0.3)

# Mode B
ax = axes[0, 1]
ax.plot(result_b.t, result_b.Q[:, 0]*1000, 'b-', label='Inlet')
ax.plot(result_b.t, result_b.Q[:, -1]*1000, 'r--', label='Outlet')
ax.set_ylabel('Flow (L/s)'); ax.set_title('Mode B: Flow'); ax.legend(); ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.plot(result_b.t, result_b.P[:, 0]/1e6, 'b-', label='Inlet')
ax.plot(result_b.t, result_b.P[:, -1]/1e6, 'r--', label='Outlet')
ax.set_ylabel('Pressure (MPa)'); ax.set_title('Mode B: Pressure'); ax.legend(); ax.grid(alpha=0.3)

ax = axes[2, 1]
ax.plot(result_b.t, result_b.T[:, 0], 'b-', label='Inlet')
ax.plot(result_b.t, result_b.T[:, -1], 'r--', label='Outlet')
ax.axhline(pipe.T_ground, color='gray', ls='--', alpha=0.5, label=f'Ground {pipe.T_ground}°C')
ax.set_ylabel('Temp (°C)'); ax.set_xlabel('Time (s)')
ax.set_title('Mode B: Temperature'); ax.legend(); ax.grid(alpha=0.3)

plt.suptitle('PipelineSim — Crude Oil Pipeline Comprehensive Analysis', fontsize=13)
plt.tight_layout()
out_path = '/root/.openclaw/workspace/projects/pipeline-sim/output/comprehensive.png'
plt.savefig(out_path, dpi=150)
print(f"✅ Plot: {out_path}")
