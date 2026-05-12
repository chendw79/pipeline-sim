#!/usr/bin/env python3
"""
Demo: Valve Closure (Water Hammer)
Runs MOC solver for rapid valve closure, saves CSV + plot.
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sim.fluid import Liquid
from sim.pipe import Pipe
from sim.solver import flow_inlet, pressure_outlet, SinglePhaseTransientSolver
from sim.steady import SteadyStateCalculator

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'demo')
os.makedirs(OUT_DIR, exist_ok=True)

# Config
pipe = Pipe(10000, 0.5, 0.012)
liquid = Liquid()

def inlet_Q(t):
    if t < 2.0: return 0.2
    elif t < 4.0: return 0.2 * (1 - (t - 2.0) / 2.0)
    else: return 0.0

# Solve
solver = SinglePhaseTransientSolver(pipe, liquid, Nx=30)
steady = SteadyStateCalculator(pipe, liquid)
V0, P0, Tp = steady.initialize_transient(0.2, 20.0, 1e6, solver)

r = solver.solve(t_max=30.0,
    inlet_bc=flow_inlet(inlet_Q, lambda t: 20.0),
    outlet_bc=pressure_outlet(lambda t: 1.0e6),
    mode='A', V_initial=V0, T_initial=20.0, P_initial=1e6)

# Save CSV
header = 't(s),x(m),P(Pa),T(C),V(m/s),Q(m3/s)'
np.savetxt(os.path.join(OUT_DIR, 'valve_closure.csv'),
    np.column_stack([
        np.repeat(r.t, r.Nx),
        np.tile(r.x, r.Nt),
        r.P.flatten(), r.T.flatten(), r.V.flatten(), r.Q.flatten()
    ]),
    delimiter=',', header=header, comments='')

# Save plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor='#1a1a2e')
fig.suptitle('Valve Closure - Water Hammer', fontsize=14, color='#00d4ff')

colors = ['#00d4ff', '#ff6b6b', '#ffd93d']
for idx, t_show in enumerate([5, 10, 15, 20]):
    ax = axes[idx//2][idx%2]
    ax.set_facecolor('#16213e')
    ax.tick_params(colors='#aaa')
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    
    i = np.argmin(np.abs(r.t - t_show))
    ax.plot(r.x/1000, r.P[i]/1e6, color=colors[idx % 3], lw=2)
    ax.set_title(f't={t_show}s', color='#ccc')
    ax.set_xlabel('Distance (km)', color='#aaa')
    ax.set_ylabel('Pressure (MPa)', color='#aaa')
    ax.grid(True, alpha=0.15)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'valve_closure.png'), dpi=150)
plt.close()

# JSON summary
summary = {
    'case': 'Valve Closure',
    'Pmax_MPa': float(r.P.max()/1e6),
    't_peak_s': float(r.t[np.argmax(r.P.max(axis=1))]),
    'P_outlet_MPa': 1.0,
    'wave_speed_mps': pipe.wave_speed(liquid),
}
with open(os.path.join(OUT_DIR, 'valve_closure_results.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f'✅ Valve closure demo: Pmax={summary["Pmax_MPa"]:.3f} MPa')
print(f'   CSV: output/demo/valve_closure.csv')
print(f'   Plot: output/demo/valve_closure.png')
