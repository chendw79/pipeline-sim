"""
pipe.py — Pipe geometry and wall properties
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List

from .fluid import Liquid

@dataclass
class Pipe:
    """Pipe segment"""
    length: float          # m
    diameter: float        # m (inner)
    wall_thickness: float  # m
    roughness: float = 4.5e-5   # m (absolute, commercial steel)
    young_modulus: float = 2.07e11  # Pa (steel)
    poisson_ratio: float = 0.3
    
    # Thermal
    heat_transfer_coeff: float = 10.0  # W/(m²·K) overall to surroundings
    T_ground: float = 10.0             # °C, ambient/ground temperature
    
    # Elevation profile (linear by default)
    elevation_start: float = 0.0  # m
    elevation_end: float = 0.0    # m
    
    def area(self) -> float:
        return np.pi * self.diameter**2 / 4.0
    
    def perimeter(self) -> float:
        return np.pi * self.diameter
    
    def wave_speed(self, liquid: Liquid) -> float:
        """Water hammer wave speed"""
        K = liquid.bulk_modulus
        rho = liquid.rho_ref
        D = self.diameter
        e = self.wall_thickness
        E = self.young_modulus
        mu = self.poisson_ratio
        # Pipe constrained at both ends
        C = 1.0 - mu**2 / 2.0
        denom = 1.0 + (K * D * C) / (E * e)
        return np.sqrt(K / rho) / np.sqrt(denom)
    
    def friction_factor(self, V: np.ndarray, Re: np.ndarray) -> np.ndarray:
        """Darcy-Weisbach friction factor"""
        f = np.zeros_like(V)
        eps = self.roughness / self.diameter
        
        # Laminar
        laminar = Re < 2000
        f[laminar] = 64.0 / np.maximum(Re[laminar], 1e-10)
        
        # Turbulent (Swamee-Jain explicit)
        turbulent = Re >= 2000
        if np.any(turbulent):
            log_arg = eps / 3.7 + 5.74 / Re[turbulent]**0.9
            f[turbulent] = 0.25 / (np.log10(np.maximum(log_arg, 1e-10)))**2
        
        return f
    
    def elevation(self, x: np.ndarray) -> np.ndarray:
        """Elevation along pipe"""
        return self.elevation_start + (self.elevation_end - self.elevation_start) * x / self.length
