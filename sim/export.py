"""
Export utilities for PipelineSim results.
"""

import os, csv, sys
import numpy as np
from .solver import TransientResult
from .pipe import Pipe
from .fluid import Liquid


def export_to_csv(result: TransientResult, filename: str, pipe: Pipe):
    """
    Export simulation results to CSV file.
    
    Format: each row = one timestep, columns = time + (P,T,V,Q) at each node
    """
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['time_s']
        for i in range(len(result.x)):
            header += [f'P_{i}_Pa', f'T_{i}_C', f'V_{i}_mps', f'Q_{i}_m3s']
        writer.writerow(header)
        
        # Write sampled time steps (max 1000 rows)
        step = max(1, result.Nt // 500)
        for n in range(0, result.Nt, step):
            row = [round(result.t[n], 4)]
            for i in range(len(result.x)):
                row += [
                    round(result.P[n, i], 2),
                    round(result.T[n, i], 4),
                    round(result.V[n, i], 6),
                    round(result.Q[n, i], 6),
                ]
            writer.writerow(row)
    print(f"  📄 CSV: {filename} ({os.path.getsize(filename)/1024:.0f} KB)")


def generate_report(
    result: TransientResult, 
    pipe: Pipe, 
    liquid: Liquid, 
    solver, 
    case_name: str
) -> str:
    """Generate a self-diagnostic report."""
    V0_initial = result.V[0, 0]
    a = solver.a
    dP_jouk = liquid.rho_ref * a * V0_initial if hasattr(solver, 'a') else 0
    
    peak_P = np.max(result.P)
    min_P = np.min(result.P)
    steady_P_end = np.mean(result.P[-5:, -1])
    
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
│  CFL = {solver.dt * a / solver.dx:.2f} (1.0 = optimal)
│  Duration: {result.t[-1]:.1f}s, Steps: {result.Nt}
│  
├─ Results Overview
│  Pressure range: {min_P/1e6:.2f} ~ {peak_P/1e6:.2f} MPa
│  Temperature range: {np.min(result.T):.1f} ~ {np.max(result.T):.1f} °C
│  Velocity range: {np.min(result.V):.4f} ~ {np.max(result.V):.4f} m/s
│  
├─ Validation
│  Initial velocity V₀ = {V0_initial:.4f} m/s
│  Joukowsky ΔP = {dP_jouk/1e6:.4f} MPa
│  
└─ Stability Check
   Final pressure std (last 10%): {np.std(result.P[-max(5,result.Nt//10):, -1]):.1f} Pa
   {'✅ NUMERICALLY STABLE' if np.std(result.P[-max(5,result.Nt//10):, -1]) < 1e5 else '⚠️ OSCILLATIONS DETECTED'}
"""
    return report


def export_hdf5(result: TransientResult, filename: str):
    """
    Export to HDF5 format (if h5py available).
    More efficient than CSV for large datasets.
    """
    try:
        import h5py
        with h5py.File(filename, 'w') as f:
            f.create_dataset('time', data=result.t)
            f.create_dataset('x', data=result.x)
            f.create_dataset('pressure', data=result.P)
            f.create_dataset('temperature', data=result.T)
            f.create_dataset('velocity', data=result.V)
            f.create_dataset('flow_rate', data=result.Q)
        print(f"  📄 HDF5: {filename} ({os.path.getsize(filename)/1024:.0f} KB)")
    except ImportError:
        print("  ⚠️ h5py not available, skipping HDF5 export")


def export_json_summary(result: TransientResult, filename: str, pipe: Pipe):
    """Export summary statistics as JSON."""
    import json
    
    summary = {
        'case': os.path.basename(filename).replace('_summary.json', ''),
        'pipe_length_m': float(pipe.length),
        'pipe_diameter_m': float(pipe.diameter),
        'simulation_time_s': float(result.t[-1]),
        'time_steps': int(result.Nt),
        'grid_points': int(len(result.x)),
        'pressure_min_Pa': float(np.min(result.P)),
        'pressure_max_Pa': float(np.max(result.P)),
        'pressure_final_Pa': float(np.mean(result.P[-5:, -1])),
        'temperature_min_C': float(np.min(result.T)),
        'temperature_max_C': float(np.max(result.T)),
    }
    
    with open(filename, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  📄 JSON: {filename}")
