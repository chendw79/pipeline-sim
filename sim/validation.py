"""
PipelineSim — Input validation module.

Validates all user inputs before simulation to provide 
clear, actionable error messages instead of cryptic 
NumPy tracebacks.
"""

from .pipe import Pipe
from .fluid import Liquid
from typing import Callable, Optional
import numpy as np


class ValidationError(Exception):
    """Custom exception for input validation errors."""
    pass


def validate_pipe(pipe: Pipe) -> None:
    """Validate pipe parameters."""
    errors = []
    
    if pipe.length <= 0:
        errors.append(f"pipe.length = {pipe.length} must be > 0")
    if pipe.length > 1e8:
        errors.append(f"pipe.length = {pipe.length/1000:.0f}km unreasonably long")
    
    if pipe.diameter <= 0:
        errors.append(f"pipe.diameter = {pipe.diameter} must be > 0")
    if pipe.diameter > 10:
        errors.append(f"pipe.diameter = {pipe.diameter}m unreasonably large")
    
    if pipe.wall_thickness <= 0:
        errors.append(f"pipe.wall_thickness = {pipe.wall_thickness} must be > 0")
    if pipe.wall_thickness > pipe.diameter:
        errors.append(f"wall_thickness ({pipe.wall_thickness}) > diameter ({pipe.diameter})")
    
    if pipe.roughness <= 0:
        errors.append(f"pipe.roughness = {pipe.roughness} must be > 0")
    
    if pipe.young_modulus <= 0:
        errors.append(f"pipe.young_modulus = {pipe.young_modulus} must be > 0")
    
    if pipe.heat_transfer_coeff < 0:
        errors.append(f"heat_transfer_coeff = {pipe.heat_transfer_coeff} cannot be negative")
    
    if np.isnan(pipe.length) or np.isinf(pipe.length):
        errors.append("pipe.length is NaN or Inf")
    
    if errors:
        raise ValidationError("Pipe validation failed:\n  " + "\n  ".join(errors))


def validate_liquid(liquid: Liquid) -> None:
    """Validate fluid parameters."""
    errors = []
    
    if liquid.rho_ref <= 0:
        errors.append(f"rho_ref = {liquid.rho_ref} must be > 0")
    if liquid.rho_ref > 20000:
        errors.append(f"rho_ref = {liquid.rho_ref} unreasonably large")
    
    if liquid.bulk_modulus <= 0:
        errors.append(f"bulk_modulus = {liquid.bulk_modulus/1e9:.2f} GPa must be > 0")
    if liquid.bulk_modulus > 1e12:
        errors.append(f"bulk_modulus = {liquid.bulk_modulus/1e9:.2f} GPa unreasonably large")
    
    if liquid.cp <= 0:
        errors.append(f"cp = {liquid.cp} must be > 0")
    
    if liquid.viscosity_ref <= 0:
        errors.append(f"viscosity_ref = {liquid.viscosity_ref} must be > 0")
    if liquid.viscosity_ref > 1000:
        errors.append(f"viscosity_ref = {liquid.viscosity_ref} Pa·s unreasonably large")
    
    if errors:
        raise ValidationError("Fluid validation failed:\n  " + "\n  ".join(errors))


def validate_solver_params(
    Nx: int,
    t_max: float,
    V_initial: float,
    mode: str,
    P_initial=None,
    T_initial=None,
) -> None:
    """Validate solver parameters."""
    errors = []
    
    if Nx < 3:
        errors.append(f"Nx = {Nx} must be >= 3")
    if Nx > 10000:
        errors.append(f"Nx = {Nx} too large, will exceed memory")
    
    if t_max <= 0:
        errors.append(f"t_max = {t_max} must be > 0")
    if t_max > 1e7:
        errors.append(f"t_max = {t_max/86400:.1f} days unreasonably long")
    
    if V_initial < 0 or V_initial > 100:
        errors.append(f"V_initial = {V_initial} m/s outside typical range [0, 100]")
    
    if mode.upper() not in ['A', 'B']:
        errors.append(f"mode = '{mode}' must be 'A' or 'B'")
    
    if P_initial is not None:
        if np.any(P_initial < 0) and np.any(P_initial < -1e5):
            # Allow small negative from gauge pressure
            errors.append(f"P_initial has values < -1e5 Pa: min={np.min(P_initial)/1e6:.2f} MPa")
    
    if T_initial is not None:
        if np.any(T_initial < -273):
            errors.append(f"T_initial below absolute zero: min={np.min(T_initial):.1f}°C")
        if np.any(T_initial > 2000):
            errors.append(f"T_initial > 2000°C: max={np.max(T_initial):.1f}°C")
    
    if errors:
        raise ValidationError("Solver parameter validation failed:\n  " + "\n  ".join(errors))


def validate_steady_params(Q: float, P_outlet: float, T_inlet: float) -> None:
    """Validate steady-state calculator parameters."""
    errors = []
    
    if Q <= 0:
        errors.append(f"flow Q = {Q} must be > 0 for steady state")
    if Q > 100:
        errors.append(f"flow Q = {Q*1000:.0f} L/s unreasonably large")
    
    if P_outlet < 0:
        errors.append(f"P_outlet = {P_outlet/1e6:.2f} MPa < 0")
    
    if T_inlet > 500:
        errors.append(f"T_inlet = {T_inlet:.0f}°C unreasonably high")
    if T_inlet < -50:
        errors.append(f"T_inlet = {T_inlet:.0f}°C unreasonably low")
    
    if errors:
        raise ValidationError("Steady-state validation failed:\n  " + "\n  ".join(errors))


def safe_solve(solver, **kwargs):
    """
    Wrapper that validates inputs before calling solver.solve().
    Provides helpful error messages on failure.
    """
    from .solver import SinglePhaseTransientSolver
    from .pipe import Pipe
    from .fluid import Liquid
    
    if not isinstance(solver, SinglePhaseTransientSolver):
        raise ValidationError(f"solver must be SinglePhaseTransientSolver, got {type(solver).__name__}")
    
    # Validate pipe and liquid
    validate_pipe(solver.pipe)
    validate_liquid(solver.liquid)
    
    # Validate solver params
    validate_solver_params(
        Nx=solver.Nx,
        t_max=kwargs.get('t_max', 0),
        V_initial=kwargs.get('V_initial', 0),
        mode=kwargs.get('mode', 'A'),
        P_initial=kwargs.get('P_initial'),
        T_initial=kwargs.get('T_initial'),
    )
    
    return solver.solve(**kwargs)
