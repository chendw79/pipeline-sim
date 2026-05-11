"""
PipelineNetwork — Multi-pipe network model.

Series connection of pipe segments. Each segment shares the same
flow rate (continuity), but has independent pressure/temperature
profiles determined by segment geometry and elevation.

Approach:
1. Steady-state analysis: sequential segments share flow, cumulative P drop
2. Transient: each segment solved independently with Mode A,
   using segment-specific outlet pressure (cumulative from downstream)

Temperature coupling: inlet temp of segment N+1 = outlet temp of segment N.
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass

from .pipe import Pipe
from .fluid import Liquid
from .solver import (
    SinglePhaseTransientSolver, TransientResult,
    flow_inlet, pressure_outlet,
)
from .steady import SteadyStateCalculator


@dataclass
class PipeSegment:
    """A single pipe segment in a multi-pipe network."""
    pipe: Pipe
    liquid: Liquid
    Nx: int = 40
    name: str = ""
    
    def __post_init__(self):
        if not self.name:
            self.name = f"Pipe-{id(self):04x}"


class SeriesNetwork:
    """
    Series-connected pipeline network.
    
    Example:
        >>> seg1 = PipeSegment(Pipe(10000, 0.6, 0.014), liquid)
        >>> seg2 = PipeSegment(Pipe(8000, 0.5, 0.012), liquid)
        >>> net = SeriesNetwork([seg1, seg2])
        >>> results = net.solve(60.0, Q_inlet=0.2, P_outlet=1e6, T_inlet=45.0)
    """
    
    def __init__(self, segments: List[PipeSegment]):
        if len(segments) < 1:
            raise ValueError("Need at least one pipe segment")
        self.segments = segments
        self.liquid = segments[0].liquid
        self.calculators = [
            SteadyStateCalculator(seg.pipe, seg.liquid) 
            for seg in segments
        ]
    
    @property
    def total_length(self) -> float:
        return sum(seg.pipe.length for seg in self.segments)
    
    def compute_outlet_pressures(
        self, Q: float, global_P_out: float
    ) -> List[float]:
        """
        Compute outlet pressure for each segment in a series network.
        
        Works backwards: last segment gets global_P_out,
        then we add cumulative P drop of downstream segments.
        
        Returns list where P_out[i] = pressure at outlet of segment i.
        """
        P_outs = [0.0] * len(self.segments)
        P_outs[-1] = global_P_out
        
        # Work backwards from second-to-last segment
        for i in range(len(self.segments) - 2, -1, -1):
            # Pressure drop of segment i+1 at steady state
            _, P = self.calculators[i+1].pressure_profile(Q, P_outs[i+1], 2)
            P_outs[i] = P[0]  # Inlet of next segment = outlet of segment i
        
        return P_outs
    
    def steady_state(
        self, Q: float, P_outlet: float, T_inlet: float,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Compute steady-state across all segments.
        
        Returns (P_profiles, T_profiles, x_profiles) for each segment.
        """
        segment_P_outlets = self.compute_outlet_pressures(Q, P_outlet)
        
        P_profiles = []
        T_profiles = []
        x_profiles = []
        cumulative_length = 0.0
        T_current = T_inlet
        
        for i, (seg, calc) in enumerate(zip(self.segments, self.calculators)):
            x_seg, P_seg = calc.pressure_profile(Q, segment_P_outlets[i], seg.Nx + 1)
            x_seg_T, T_seg = calc.temperature_profile(Q, T_current, seg.Nx + 1)
            
            # Shift x to global coordinate
            x_global = x_seg_T + cumulative_length
            
            P_profiles.append(P_seg)
            T_profiles.append(T_seg)
            x_profiles.append(x_global)
            
            cumulative_length += seg.pipe.length
            T_current = T_seg[-1]
        
        return P_profiles, T_profiles, x_profiles
    
    def solve(
        self,
        t_max: float,
        Q_inlet: float,
        P_outlet: float,
        T_inlet: float = 20.0,
        verbose: bool = True,
    ) -> List[TransientResult]:
        """
        Solve transient for all segments in series.
        
        All segments share the same flow rate Q_inlet.
        Temperature couples sequentially (outlet N → inlet N+1).
        Pressure drop is cumulative across segments.
        """
        results = []
        T_current = T_inlet
        segment_P_outlets = self.compute_outlet_pressures(Q_inlet, P_outlet)
        
        for i, (seg, calc) in enumerate(zip(self.segments, self.calculators)):
            is_last = (i == len(self.segments) - 1)
            
            if verbose:
                print(f"\n  Segment {i+1}/{len(self.segments)}: {seg.name} "
                      f"(L={seg.pipe.length/1000:.1f}km, D={seg.pipe.diameter*1000:.0f}mm)")
            
            solver = SinglePhaseTransientSolver(seg.pipe, seg.liquid, Nx=seg.Nx)
            
            # Steady-state initialization
            V0 = Q_inlet / seg.pipe.area()
            _, P_init = calc.pressure_profile(Q_inlet, segment_P_outlets[i], seg.Nx + 1)
            _, T_init = calc.temperature_profile(Q_inlet, T_current, seg.Nx + 1)
            
            # Boundary conditions
            inlet_bc = flow_inlet(lambda t: Q_inlet, lambda t: T_current)
            outlet_bc = pressure_outlet(lambda t: segment_P_outlets[i])
            
            result = solver.solve(
                t_max=t_max,
                inlet_bc=inlet_bc,
                outlet_bc=outlet_bc,
                mode='A',
                V_initial=V0,
                P_initial=P_init,
                T_initial=T_init,
            )
            
            results.append(result)
            T_current = result.T[-1, -1]
            
            if verbose:
                P_range = (np.min(result.P)/1e6, np.max(result.P)/1e6)
                T_range = (np.min(result.T), np.max(result.T))
                print(f"    P: {P_range[0]:.2f}~{P_range[1]:.2f} MPa, "
                      f"T: {T_range[0]:.1f}~{T_range[1]:.1f}°C")
        
        return results
    
    def combine_results(self, results: List[TransientResult]) -> TransientResult:
        """Combine segment results into a single composite."""
        if not results:
            raise ValueError("No results to combine")
        
        base = results[0]
        
        # Build x array
        x_total = np.concatenate([r.x for r in results])
        Nx_total = len(x_total)
        
        # Interpolate segments whose Nt differs to match the longest
        max_Nt = max(r.Nt for r in results)
        t_unified = base.t if base.Nt == max_Nt else None
        
        # Need to find common time base
        # All use same CFL so Nt should be close - pick the largest
        unified_results = []
        for r in results:
            if r.Nt != max_Nt:
                # Interpolate using numpy (no scipy dependency)
                P_new = np.zeros((max_Nt, r.x.shape[0]))
                T_new = np.zeros((max_Nt, r.x.shape[0]))
                V_new = np.zeros((max_Nt, r.x.shape[0]))
                Q_new = np.zeros((max_Nt, r.x.shape[0]))
                if max_Nt > 1 and r.Nt > 1:
                    t_old = r.t
                    t_new = np.linspace(t_old[0], t_old[-1], max_Nt)
                    for j in range(r.x.shape[0]):
                        P_new[:, j] = np.interp(t_new, t_old, r.P[:, j])
                        T_new[:, j] = np.interp(t_new, t_old, r.T[:, j])
                        V_new[:, j] = np.interp(t_new, t_old, r.V[:, j])
                        Q_new[:, j] = np.interp(t_new, t_old, r.Q[:, j])
                unified_results.append(TransientResult(
                    t=t_new if max_Nt > 1 else r.t,
                    x=r.x, P=P_new, T=T_new, V=V_new, Q=Q_new,
                    rho=r.rho, mu=r.mu,
                ))
            else:
                unified_results.append(r)
        
        # Concatenate
        P_total = np.concatenate([r.P for r in unified_results], axis=1)
        T_total = np.concatenate([r.T for r in unified_results], axis=1)
        V_total = np.concatenate([r.V for r in unified_results], axis=1)
        Q_total = np.concatenate([r.Q for r in unified_results], axis=1)
        
        return TransientResult(
            t=base.t.copy() if max_Nt == base.Nt else t_new,
            x=x_total, P=P_total, T=T_total, V=V_total, Q=Q_total,
            rho=results[0].rho.copy(),
            mu=results[0].mu.copy(),
        )
