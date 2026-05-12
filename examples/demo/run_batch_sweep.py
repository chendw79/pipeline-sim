#!/usr/bin/env python3
"""
Demo: Parameter Sweep (3 flow rates)
Compare Joukowsky pressure rise for different initial velocities.
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sim.fluid import Liquid
from sim.pipe import Pipe
from sim.solver import flow_inlet, pressure_outlet, SinglePhaseTransientSolver
from sim.steady import SteadyStateCalculator
from sim.solver_advanced import MacCormackSolver

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'demo')
os.makedirs(OUT_DIR, exist_ok=True)

pipe = Pipe(10000, 0.5, 0.012)
liquid = Liquid()
a = pipe.wave_speed(liquid)

flow_rates = [0.1, 0.2, 0.3]
results_data = []

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1a1a2e')
ax.set_facecolor('#16213e')
colors = ['#00d4ff', '#ffd93d', '#ff6b6b']

for qi, Q0 in enumerate(flow_rates):
    def inlet_Q(t, q0=Q0):
        if t < 2.0: return q0
        else: return 0.0
    
    solver = SinglePhaseTransientSolver(pipe, liquid, Nx=30)
    steady = SteadyStateCalculator(pipe, liquid)
    V0_init, P0_init, Tp = steady.initialize_transient(Q0, 20.0, 1e6, solver)
    
    r = solver.solve(t_max=30.0,
        inlet_bc=flow_inlet(inlet_Q, lambda t: 20.0),
        outlet_bc=pressure_outlet(lambda t: 1.0e6),
        mode='A', V_initial=V0_init, T_initial=20.0, P_initial=1e6)
    
    Pmax = r.P.max()/1e6
    V_init = Q0 / pipe.area()
    Joukowsky = 1.0 + 1000 * a * V_init / 1e6
    results_data.append([Q0, V_init, Pmax, Joukowsky])
    
    ax.plot(r.t, r.P[:,0]/1e6, color=colors[qi], lw=2,
            label=f'Q₀={Q0} m³/s (V={V_init:.2f}m/s) Pmax={Pmax:.2f}MPa')

ax.axhline(1.0, color='#666', ls='--', alpha=0.5)
ax.set_xlabel('Time (s)', color='#aaa')
ax.set_ylabel('Inlet Pressure (MPa)', color='#aaa')
ax.set_title('Parameter Sweep: Joukowsky Pressure Rise by Flow Rate', color='#00d4ff')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.15)
ax.tick_params(colors='#aaa')
for spine in ax.spines.values(): spine.set_color('#333')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'batch_sweep.png'), dpi=150)
plt.close()

# Save CSV
np.savetxt(os.path.join(OUT_DIR, 'batch_sweep.csv'),
    np.array(results_data),
    delimiter=',',
    header='Q0(m3/s),V0(m/s),Pmax_MOC(MPa),Joukowsky_est(MPa)', comments='')

summary = {
    'case': 'Parameter Sweep',
    'wave_speed_mps': pipe.wave_speed(liquid),
    'results': [
        {'Q0_m3s': Q0, 'V0_ms': V0, 'Pmax_MPa': pmax, 'Joukowsky_MPa': jk}
        for Q0, V0, pmax, jk in results_data
    ],
}
with open(os.path.join(OUT_DIR, 'batch_sweep_results.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f'✅ Batch sweep demo:')
for Q0, V0, pmax, jk in results_data:
    ratio = (pmax-1.0)/(jk-1.0)*100 if jk > 1.0 else 0
    print(f'   Q₀={Q0}: Pmax={pmax:.3f} MPa (Joukowsky={jk:.3f} MPa, {ratio:.0f}%)')
