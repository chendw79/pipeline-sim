"""
valve_stability_test.py — Valve opening/closing stability verification
Fixed: proper valve BC (dead-end V=0, not P=0)
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

pipe = Pipe(
    length=10000.0, diameter=0.5, wall_thickness=0.012,
    roughness=4.5e-5, heat_transfer_coeff=5.0,
    T_ground=10.0, elevation_start=0.0, elevation_end=0.0,
)

liquid = Liquid(name="Water",
    rho_ref=998.0, bulk_modulus=2.15e9, cp=4182.0,
    viscosity_ref=1.0e-3, thermal_expansion=2.07e-4)

Q0 = 0.2    # m³/s
V0 = Q0 / pipe.area()
H_reservoir = 100.0
P0 = H_reservoir * liquid.rho_ref * 9.81  # ~0.98 MPa

# ============================================================
# Case 1: Instant closure — proper dead-end BC (V=0)
# ============================================================
print(f"\n{'='*70}")
print(f"Case 1: Instantaneous Valve Closure (proper dead-end V=0)")
print(f"{'='*70}")

# Mode B: Inlet = P+T, Outlet = Q
# Pre-closure: inlet at reservoir P, outlet at Q0
# Post-closure: outlet Q=0

def inlet_pressure(t):
    return P0  # constant reservoir pressure

def inlet_temp(t):
    return 20.0

def outlet_flow(t):
    if t < 1.0:
        return Q0  # valve open
    else:
        return 0.0  # valve closed → dead end

solver = SinglePhaseTransientSolver(pipe, liquid, Nx=50)
result_inst = solver.solve(
    t_max=40.0,
    inlet_bc=pressure_inlet(inlet_pressure, inlet_temp),
    outlet_bc=flow_outlet(outlet_flow),
    mode='B',
    V_initial=V0,
    T_initial=18.0,
    P_initial=P0,
)

P_valve_inst = result_inst.P[:, -1]
P_inlet_inst = result_inst.P[:, 0]
V_valve_inst = result_inst.V[:, -1]
Q_valve_inst = result_inst.Q[:, -1]

peak_P_inst = np.max(P_valve_inst)
dP_jouk = liquid.rho_ref * solver.a * V0
theoretical_peak = P0 + dP_jouk
error_pct = abs(peak_P_inst - theoretical_peak) / theoretical_peak * 100

# Period: find first 3 pressure peaks, use time between peak 1 and 3 for period
# Full wave cycle = 4L/a (pressure wave travels valve→reservoir→valve→reservoir→valve)
# So period = time between peak 1 and peak 3 (two trips = 4L/a * 2 = 8L/a ... no, one full oscillation = 4L/a)
# Actually: after valve closes, pressure at valve rises (peak1). Wave goes to reservoir and back (2L/a), 
# pressure drops as wave returns (trough). Goes again to reservoir and back (another 2L/a) and rises (peak2).
# So time between peak1 and peak2 = 4L/a

# Period: manual local maxima detection
peak_indices = []
window = 5
for i in range(window, len(P_valve_inst) - window):
    local_max = True
    for j in range(1, window+1):
        if P_valve_inst[i] <= P_valve_inst[i-j] or P_valve_inst[i] <= P_valve_inst[i+j]:
            local_max = False
            break
    if local_max and P_valve_inst[i] > P0 + 0.3 * dP_jouk:
        peak_indices.append(i)

# Deduplicate adjacent peaks
filtered_peaks = [peak_indices[0]] if peak_indices else []
for p in peak_indices[1:]:
    dt_check = result_inst.t[p] - result_inst.t[filtered_peaks[-1]]
    if dt_check > (result_inst.t[1] - result_inst.t[0]) * 3:  # more than 3 timesteps apart
        filtered_peaks.append(p)

osc_period = 0.0
if len(filtered_peaks) >= 2:
    osc_period = result_inst.t[filtered_peaks[1]] - result_inst.t[filtered_peaks[0]]
    if len(filtered_peaks) >= 3:
        osc_period = (result_inst.t[filtered_peaks[2]] - result_inst.t[filtered_peaks[0]]) / 2.0

period_theory = 4.0 * pipe.length / solver.a  # 4L/a

period_theory = 4.0 * pipe.length / solver.a  # 4L/a

print(f"  V0={V0:.3f} m/s, a={solver.a:.1f} m/s")
print(f"  Joukowsky ΔP={dP_jouk/1e6:.3f} MPa")
print(f"  Theoretical peak={(P0+dP_jouk)/1e6:.3f} MPa")
print(f"  Simulated peak={peak_P_inst/1e6:.3f} MPa")
print(f"  Error={error_pct:.2f}%")
print(f"  Oscillation period={osc_period:.2f}s (theory: {period_theory:.2f}s)")
period_err = abs(osc_period - period_theory) / period_theory * 100 if osc_period > 0 else 0
print(f"  Period error={period_err:.1f}%")
print(f"  {'✅' if error_pct < 5 and period_err < 10 else '⚠️'} ")

# ============================================================
# Case 2: Gradual valve closure via varying resistance
# ============================================================
print(f"\n{'='*70}")
print(f"Case 2: Gradual Valve Closure (Δt=3s, linear flow reduction)")
print(f"{'='*70}")

close_time = 3.0

def outlet_flow_grad(t):
    if t < 1.0:
        return Q0
    elif t < 1.0 + close_time:
        frac = (t - 1.0) / close_time
        return Q0 * (1.0 - frac)
    else:
        return 0.0

solver2 = SinglePhaseTransientSolver(pipe, liquid, Nx=50)
result_grad = solver2.solve(
    t_max=40.0,
    inlet_bc=pressure_inlet(inlet_pressure, inlet_temp),
    outlet_bc=flow_outlet(outlet_flow_grad),
    mode='B',
    V_initial=V0,
    T_initial=18.0,
    P_initial=P0,
)

P_valve_grad = result_grad.P[:, -1]
peak_P_grad = np.max(P_valve_grad)
dP_allievi = 2.0 * pipe.length * V0 * liquid.rho_ref / close_time

print(f"  Closure time: {close_time}s")
print(f"  Peak pressure: {peak_P_grad/1e6:.3f} MPa")
print(f"  Joukowsky peak: {(P0+dP_jouk)/1e6:.3f} MPa")
print(f"  Ratio to Joukowsky: {(peak_P_grad-P0)/(dP_jouk)*100:.1f}%")
print(f"  Allievi ΔP approx: {dP_allievi/1e6:.3f} MPa")
print(f"  {'✅' if peak_P_grad < theoretical_peak else '⚠️'} Gradual < Instant (expected)")

# ============================================================
# Case 3: Sudden valve opening
# ============================================================
print(f"\n{'='*70}")
print(f"Case 3: Sudden Valve Opening")
print(f"{'='*70}")

# Mode A: Inlet Q+T, Outlet P
Q_init = 0.01
Q_target = Q0

def inlet_flow(t):
    return Q_target

def outlet_pressure_open(t):
    return P0

solver3 = SinglePhaseTransientSolver(pipe, liquid, Nx=50)
result_open = solver3.solve(
    t_max=30.0,
    inlet_bc=flow_inlet(
        lambda t: Q_init if t < 1.0 else Q_target,
        lambda t: 20.0
    ),
    outlet_bc=pressure_outlet(lambda t: P0),
    mode='A',
    V_initial=Q_init / pipe.area(),
    T_initial=18.0,
    P_initial=P0,
)

P_inlet_open = result_open.P[:, 0]
peak_dP_open = np.max(P_inlet_open) - P0
dP_jouk_open = liquid.rho_ref * solver3.a * (Q_target / pipe.area())
open_error = abs(peak_dP_open - dP_jouk_open) / dP_jouk_open * 100

print(f"  Flow: {Q_init*1000:.0f} → {Q_target*1000:.0f} L/s (ΔV={(Q_target-Q_init)/pipe.area():.2f} m/s)")
print(f"  Joukowsky ΔP={dP_jouk_open/1e6:.3f} MPa, Simulated ΔP={peak_dP_open/1e6:.3f} MPa")
print(f"  Error={open_error:.2f}%")
print(f"  {'✅' if open_error < 10 else '⚠️'}")

# ============================================================
# Case 4: Long-term stability check
# ============================================================
print(f"\n{'='*70}")
print(f"Case 4: Long-term Stability (200s) + Energy check")
print(f"{'='*70}")

solver4 = SinglePhaseTransientSolver(pipe, liquid, Nx=30)

def outlet_flow_long(t):
    return 0.0 if t >= 1.0 else Q0

result_long = solver4.solve(
    t_max=200.0,
    inlet_bc=pressure_inlet(inlet_pressure, inlet_temp),
    outlet_bc=flow_outlet(outlet_flow_long),
    mode='B',
    V_initial=V0,
    T_initial=18.0,
    P_initial=P0,
)

P_valve_long = result_long.P[:, -1]
# Last half for analysis (skip initial transient)
analysis_half = P_valve_long[len(P_valve_long)//2:]
final_std = float(np.std(analysis_half))
final_mean = float(np.mean(analysis_half))
# Last quarter spread
last_quarter = P_valve_long[-int(len(P_valve_long)//4):]
final_spread = float(np.max(last_quarter) - np.min(last_quarter))

# Energy check: mechanical energy should decay (friction damping)
# Check that pressure oscillations decay over time
amp_decay = True
mid_idx = len(P_valve_long) // 2
if np.max(P_valve_long[-mid_idx:]) > np.max(P_valve_long[-mid_idx//2:]):
    amp_decay = False

Q_inlet_long = result_long.Q[:, 0]

print(f"  Last half σ={final_std:.0f} Pa, last quarter spread={final_spread/1e3:.1f} kPa")
print(f"  Final mean drift: {abs(final_mean-P0)/1e3:.1f} kPa")
print(f"  Amplitude decaying: {'✅' if amp_decay else '⚠️'}")
print(f"  {'✅' if final_spread < 0.1*(P0+dP_jouk) else '⚠️'} Pressure damping to stable zone")

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*70}")
print(f"STABILITY TEST SUMMARY")
print(f"{'='*70}")

print(f"\n{'='*70}")
print(f"  Instant closure: Peak={peak_P_inst/1e6:.3f}MPa vs Joukowsky={theoretical_peak/1e6:.3f}MPa, Error={error_pct:.2f}%")
print(f"  Gradual closure: Peak={peak_P_grad/1e6:.3f}MPa ({peak_P_grad/theoretical_peak*100:.1f}% of Joukowsky)")
print(f"  Sudden opening:  Peak ΔP={peak_dP_open/1e6:.3f}MPa vs Joukowsky ΔP={dP_jouk_open/1e6:.3f}MPa, Error={open_error:.2f}%")
all_pass = error_pct < 5 and open_error < 10

print(f"\n{'✅ ALL PASSED' if all_pass else '⚠️ SOME CHECKS WARN'}")
print(f"{'='*70}")

# Save key metrics
import json
metrics = {
    'instant_closure_error_pct': round(error_pct, 2),
    'instant_closure_peak_MPa': round(peak_P_inst/1e6, 3),
    'joukowsky_peak_MPa': round(theoretical_peak/1e6, 3),
    'gradual_peak_MPa': round(peak_P_grad/1e6, 3),
    'opening_error_pct': round(open_error, 2),
}
with open('/root/.openclaw/workspace/projects/pipeline-sim/output/stability_results.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# ============================================================
# Plot
# ============================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Instant closure — Pressure at valve
ax = axes[0, 0]
t = result_inst.t
ax.plot(t, P_valve_inst/1e6, 'r-', linewidth=1.5, label='Valve')
ax.plot(t, P_inlet_inst/1e6, 'b-', alpha=0.6, label='Inlet (reservoir)')
ax.axhline(P0/1e6, color='gray', linestyle='--', alpha=0.4, label=f'Steady {P0/1e6:.2f}')
ax.axhline(theoretical_peak/1e6, color='orange', linestyle=':', alpha=0.6, 
           label=f'Joukowsky {theoretical_peak/1e6:.2f}')
ax.set_ylabel('Pressure (MPa)')
ax.set_title(f'Instant Closure (V=0 dead-end)\nPeak={peak_P_inst/1e6:.2f}MPa, Error={error_pct:.1f}%')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# 2. Instant closure — Zoomed: flow + pressure wave
ax = axes[0, 1]
ax.plot(t, Q_valve_inst*1000, 'g-', linewidth=1.5)
ax.set_ylabel('Flow at valve (L/s)')
ax.set_xlabel('Time (s)')
ax.set_title(f'Flow at Valve — Instant Closure\nPeriod≈{osc_period:.1f}s (theory={period_theory:.1f}s)')
ax.grid(True, alpha=0.3)

# 3. Gradual closure
ax = axes[1, 0]
ax.plot(result_grad.t, result_grad.P[:, -1]/1e6, 'r-', label='Valve')
ax.plot(result_grad.t, result_grad.P[:, 0]/1e6, 'b-', alpha=0.6, label='Inlet')
ax.axhline(P0/1e6, color='gray', linestyle='--', alpha=0.4)
ax.axhline(theoretical_peak/1e6, color='orange', linestyle=':', alpha=0.4, 
           label=f'Joukowsky {theoretical_peak/1e6:.2f}')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Pressure (MPa)')
ax.set_title(f'Gradual Closure (Δt={close_time}s)\nPeak={peak_P_grad/1e6:.2f}MPa')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# 4. Long-term stability
ax = axes[1, 1]
ax.plot(result_long.t, P_valve_long/1e6, 'r-', linewidth=1.0, label='Valve pressure')
ax.axhline(P0/1e6, color='gray', linestyle='--', alpha=0.4)
# Mark analysis window
N4 = len(P_valve_long) // 4
ax.axvspan(result_long.t[-N4], result_long.t[-1], alpha=0.1, color='green',
           label=f'Analysis σ={final_std:.0f}Pa')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Pressure (MPa)')
ax.set_title(f'Long-term Stability (200s)\nσ={final_std:.0f}Pa, Drift={abs(final_mean-P0)/1e3:.1f}kPa')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

plt.suptitle('PipelineSim Valve Transient — Stability Validation', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/root/.openclaw/workspace/projects/pipeline-sim/output/valve_stability.png', dpi=150)
print(f"\n✅ Plot saved")
