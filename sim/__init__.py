"""
PipelineSim — Single-phase liquid pipeline transient simulator.

Solvers:
  - MOC (Method of Characteristics) — default, robust, CFL-constrained
  - IFDM (Implicit Finite Difference) — unconditionally stable, large timesteps
  - MacCormack (Predictor-Corrector) — 2nd order accurate, sharp wave fronts
  - FVM (Finite Volume MUSCL-Hancock) — conservative, strong shock capture
"""

__version__ = "0.4.0"

from .fluid import Liquid
from .pipe import Pipe
from .steady import SteadyStateCalculator
from .solver import (SinglePhaseTransientSolver, 
                     flow_inlet, pressure_inlet, pressure_outlet, flow_outlet,
                     step, TransientResult)
from .solver_advanced import (ImplicitFDMSolver, MacCormackSolver, 
                              FiniteVolumeSolver, compare_solvers)
from .pump import Pump
from .export import export_to_csv, generate_report

__all__ = [
    "Liquid", "Pipe", "SteadyStateCalculator",
    "SinglePhaseTransientSolver",
    "ImplicitFDMSolver", "MacCormackSolver", "FiniteVolumeSolver",
    "compare_solvers",
    "flow_inlet", "pressure_inlet", "pressure_outlet", "flow_outlet",
    "step", "TransientResult",
    "Pump",
    "export_to_csv", "generate_report",
]
