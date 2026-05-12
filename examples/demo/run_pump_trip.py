#!/usr/bin/env python3
"""
Demo: Pump Trip (Pressure Drop, Mode B)
Inlet pressure drops from 4MPa to 2MPa, constant outlet flow.
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sim.fluid import Liquid
from sim.pipe import Pipe
from sim.solver import pressure_inlet, flow_outlet, SinglePhaseTransientSolver
from sim.steady import SteadyStateCalculator

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'demo')
os.makedirs(OUT_DIR, exist_ok=True)

pipe = Pipe(20000, 0.6, 0.015)
liquid = Liquid()

def inlet_P(t):
    if t < 5.0: return 4e6
    elif t < 15.0: return 4e6 * (1 - (t - 5.0) / 10.0)
    else: return 2e6

solver = SinglePhaseTransientSolver(pipe, liquid, Nx=40)
steady = SteadyStateCalculator(pipe, liquid)
V0, P0, Tp = steady.initialize_transient(0.3, 20.0, 4e6, solver)

r = solver.solve(t_max=60.0,
    inlet_bc=pressure_inlet(inlet_P, lambda t: 20.0),
    outlet_bc=flow_outlet(lambda t: 0.3),
    mode='B', V_initial=V0, T_initial=20.0, P_initial=4e6)

# Save CSV
np.savetxt(os.path.join(OUT_DIR, 'pump_trip.csv'),
    np.column_stack([
        np.repeat(r.t, r.Nx), np.tile(r.x, r.Nt),
        r.P.flatten(), r.T.flatten(), r.V.flatten(), r.Q.flatten()
    ]),
    delimiter=',',
    header='t(s),x(m),P(Pa),T(C),V(m/s),Q(m3/s)', comments='')

# Plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(12, 8), facecolor='#1a1a2e')
fig.suptitle('Pump Trip - Inlet Pressure Drop (Mode B)', fontsize=14, color='#00d4ff')

ax = axes[0]; ax.set_facecolor('#16213e')
ax.plot(r.t, r.P[:, 0]/1e6, color='#ff6b6b', lw=2, label='Inlet')
ax.plot(r.t, r.P[:, -1]/1e6, color='#00d4ff', lw=2, label='Outlet')
ax.set_xlabel('Time (s)', color='#aaa'); ax.set_ylabel('Pressure (MPa)', color='#aaa')
ax.legend(); ax.grid(True, alpha=0.15); ax.tick_params(colors='#aaa')
for spine in ax.spines.values(): spine.set_color('#333')

ax = axes[1]; ax.set_facecolor('#16213e')
for t_show in [5, 10, 20, 40]:
    i = np.argmin(np.abs(r.t - t_show))
    ax.plot(r.x/1000, r.P[i]/1e6, lw=2, label=f't={t_show}s')
ax.set_xlabel('Distance (km)', color='#aaa'); ax.set_ylabel('Pressure (MPa)', color='#aaa')
ax.legend(); ax.grid(True, alpha=0.15); ax.tick_params(colors='#aaa')
for spine in ax.spines.values(): spine.set_color('#333')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'pump_trip.png'), dpi=150)
plt.close()

summary = {
    'case': 'Pump Trip',
    'P_inlet_final_MPa': 2.0,
    'P_min_during_transient_MPa': float(r.P[:,0].min()/1e6),
    'wave_speed_mps': pipe.wave_speed(liquid),
}
with open(os.path.join(OUT_DIR, 'pump_trip_results.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f'✅ Pump trip demo: Pmin={summary["P_min_during_transient_MPa"]:.3f} MPa')
