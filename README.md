# 🌊 PipelineSim

**Single-phase liquid pipeline transient simulator — MOC hydraulics + Finite Difference temperature coupling**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-chendw79%2Fpipeline--sim-brightgreen)](https://github.com/chendw79/pipeline-sim)

---

## Quick Start

```python
from sim.fluid import Liquid
from sim.pipe import Pipe
from sim.steady import SteadyStateCalculator
from sim.solver import SinglePhaseTransientSolver, flow_inlet, pressure_outlet

# Define pipe and fluid
pipe = Pipe(length=15000.0, diameter=0.6, wall_thickness=0.014)
liquid = Liquid(name="Crude Oil")

# Steady-state initialization (CRITICAL for realistic results)
calc = SteadyStateCalculator(pipe, liquid)
V0, P_init, T_init = calc.initialize_transient(Q=0.25, T_inlet=45.0, P_outlet=0.5e6)

# Transient simulation
solver = SinglePhaseTransientSolver(pipe, liquid, Nx=40)
result = solver.solve(
    t_max=80.0,
    inlet_bc=flow_inlet(lambda t: 0.25, lambda t: 45.0),
    outlet_bc=pressure_outlet(lambda t: 0.5e6),
    mode='A',
    V_initial=V0, P_initial=P_init, T_initial=T_init,
)

# Export
from sim.export import export_to_csv, generate_report
export_to_csv(result, 'results.csv', pipe)
print(generate_report(result, pipe, liquid, solver, "My Case"))
```

## Features

| Feature | Status |
|---------|--------|
| Method of Characteristics (MOC) hydraulics | ✅ |
| Upwind finite difference temperature | ✅ |
| Mode A: Inlet Q+T / Outlet P | ✅ |
| Mode B: Inlet P+T / Outlet Q | ✅ |
| Pressure/temperature dependent fluid properties | ✅ |
| Steady-state initialization (analytical) | ✅ |
| CSV / JSON export | ✅ |
| Self-diagnostic reports | ✅ |
| Elevation profiles | ✅ |
| Multi-pipe networks | 📋 Planned |
| Pump/valve component models | 📋 Planned |
| HDF5 export | 📋 Planned |

## Validation

**Water hammer (instant valve closure):** Peak error < 1% against Joukowsky theory.

## License

MIT
