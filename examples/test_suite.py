"""
PipelineSim test suite.

Tests:
1. Fluid properties
2. Pipe properties and wave speed
3. Steady-state calculator
4. Solver (MOC hydraulics)
5. Boundary conditions
6. Multi-pipe network
7. Pump model
8. Validation module
9. Export module
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np

from sim.fluid import Liquid
from sim.pipe import Pipe
from sim.steady import SteadyStateCalculator, analyze_pipeline
from sim.solver import (
    SinglePhaseTransientSolver,
    flow_inlet, pressure_outlet,
    pressure_inlet, flow_outlet,
)
from sim.network import PipeSegment, SeriesNetwork
from sim.pump import Pump
from sim.validation import validate_pipe, validate_liquid, ValidationError

# ============================================================
# Test 1: Fluid properties
# ============================================================
def test_fluid():
    print("Test 1: Fluid Properties")
    
    water = Liquid()
    assert water.name == "Water"
    assert water.rho_ref == 998.0
    assert water.bulk_modulus == 2.15e9
    assert water.cp == 4182.0
    
    # Density: should increase with pressure
    rho_high = water.density(1e7, 20.0)
    rho_low = water.density(0, 20.0)
    assert rho_high > rho_low, "Density should increase with pressure"
    
    # Density: should decrease with temperature
    rho_cold = water.density(1e5, 10.0)
    rho_hot = water.density(1e5, 50.0)
    assert rho_cold > rho_hot, "Density should decrease with temperature"
    
    # Viscosity: should decrease with temperature
    mu_cold = water.viscosity(10.0)
    mu_hot = water.viscosity(50.0)
    assert mu_cold > mu_hot, "Viscosity should decrease with temperature"
    
    # Viscosity: should be clipped
    mu_extreme = water.viscosity(-100)  # Below absolute zero
    assert 1e-6 <= mu_extreme <= 100.0, "Viscosity should be clipped"
    
    print("  ✅ All fluid tests passed")


# ============================================================
# Test 2: Pipe properties
# ============================================================
def test_pipe():
    print("Test 2: Pipe Properties")
    
    # Standard pipe
    pipe = Pipe(10000, 0.5, 0.012)
    assert abs(pipe.area() - 0.19635) < 1e-4, f"Area mismatch: {pipe.area()}"
    
    # Wave speed for water in steel pipe
    water = Liquid()
    a = pipe.wave_speed(water)
    assert 1000 < a < 1500, f"Wave speed out of range: {a:.0f} m/s"
    assert abs(a - 1234) < 15, f"Expected ~1234 m/s, got {a:.0f} m/s"
    
    # Crude oil in pipe
    oil = Liquid(name="Oil", rho_ref=860.0, bulk_modulus=1.8e9)
    a_oil = pipe.wave_speed(oil)
    assert 1000 < a_oil < 1500, f"Oil wave speed out of range: {a_oil:.0f}"
    
    # Friction factor
    Re = np.array([water.rho_ref * 1.0 * pipe.diameter / water.viscosity_ref])
    f = float(pipe.friction_factor(np.array([1.0]), Re)[0])
    assert 0.01 < f < 0.1, f"Friction factor out of range: {f}"
    
    print("  ✅ All pipe tests passed")


# ============================================================
# Test 3: Steady-state calculator
# ============================================================
def test_steady():
    print("Test 3: Steady-State Calculator")
    
    pipe = Pipe(15000, 0.6, 0.014)
    oil = Liquid(name="Oil", rho_ref=860.0, bulk_modulus=1.8e9, cp=2000.0)
    
    calc = SteadyStateCalculator(pipe, oil)
    
    # Pressure profile
    x, P = calc.pressure_profile(Q=0.25, P_outlet=0.5e6)
    assert len(x) == 100
    assert len(P) == 100
    assert P[0] > P[-1], "Pressure should decrease along pipe"  # positive direction
    assert P[-1] == 0.5e6, f"Outlet pressure mismatch: {P[-1]}"
    assert P[0] > 0.5e6, "Inlet pressure should be > outlet"
    
    # Temperature profile
    x, T = calc.temperature_profile(Q=0.25, T_inlet=45.0)
    assert T[0] == 45.0, f"Inlet temperature mismatch: {T[0]}"
    assert T[-1] < T[0], "Temperature should decrease along pipe"
    assert T[-1] > -50, f"Temperature went too low: {T[-1]}"
    
    # Temperature at low flow (should cool more)
    x, T_low = calc.temperature_profile(Q=0.05, T_inlet=45.0)
    assert T_low[-1] < T[-1], "Lower flow should cool more"
    
    # initialize_transient
    solver = SinglePhaseTransientSolver(pipe, oil, Nx=40)
    V0, P_init, T_init = calc.initialize_transient(0.25, 45.0, 0.5e6, solver)
    assert abs(V0 - 0.884) < 0.01, f"Velocity mismatch: {V0}"
    assert len(P_init) == 41, f"P_init length: {len(P_init)}"
    assert len(T_init) == 41
    
    # Pipeline analysis
    result = analyze_pipeline(length=15000, diameter=0.6, verbose=False)
    assert result['V'] > 0
    assert result['P_inlet'] > result['P_outlet']
    assert result['T_outlet'] < result['T_inlet']
    
    print("  ✅ All steady-state tests passed")


# ============================================================
# Test 4: Solver (MOC hydraulics)
# ============================================================
def test_solver():
    print("Test 4: Solver (MOC)")
    
    pipe = Pipe(10000, 0.5, 0.012)
    water = Liquid()
    solver = SinglePhaseTransientSolver(pipe, water, Nx=50)
    
    # Mode A: Inlet flow + outlet pressure
    result = solver.solve(
        t_max=10.0,
        inlet_bc=flow_inlet(lambda t: 0.2, lambda t: 20.0),
        outlet_bc=pressure_outlet(lambda t: 1.0e6),
        mode='A',
        V_initial=1.02, T_initial=18.0,
    )
    
    assert result.Nt > 10, f"Too few time steps: {result.Nt}"
    assert np.max(result.P) > 0, "Pressure should be positive"
    assert np.min(result.T) > -50, f"Temperature went too low: {np.min(result.T)}"
    assert np.max(result.T) < 500, f"Temperature went too high: {np.max(result.T)}"
    
    # Flow should be approximately conserved
    Q_diff = np.mean(np.abs(result.Q[:, 0] - result.Q[:, -1]))
    assert Q_diff < 0.5, f"Flow not conserved: ΔQ = {Q_diff}"
    
    # Mode B: Inlet pressure + outlet flow
    result_b = solver.solve(
        t_max=10.0,
        inlet_bc=pressure_inlet(lambda t: 2.0e6, lambda t: 20.0),
        outlet_bc=flow_outlet(lambda t: 0.2),
        mode='B',
        V_initial=1.02, T_initial=18.0,
        P_initial=1.5e6,
    )
    
    assert result_b.Nt > 10
    assert np.max(result_b.P) > 0
    
    # Test: MOC produces pressure wave propagation
    pipe_wh = Pipe(1000, 0.5, 0.012, roughness=4.5e-5)
    solver_wh = SinglePhaseTransientSolver(pipe_wh, water, Nx=30)
    
    # Step change in inlet flow
    def inlet_Q(t):
        if t < 0.05:
            return 0.2
        return 0.15  # 25% reduction
    
    result_wh = solver_wh.solve(
        t_max=4.0,
        inlet_bc=flow_inlet(inlet_Q, lambda t: 20.0),
        outlet_bc=pressure_outlet(lambda t: 1.0e6),
        mode='A',
        V_initial=1.02, T_initial=20.0,
        P_initial=1.5e6,
    )
    
    # Check that pressure changes over time (transient behavior)
    P_std = np.std(result_wh.P[:, 0]) / 1e6
    assert P_std > 0.01, f"Transient behavior too weak: P_std={P_std:.4f} MPa"
    print("  ✅ All solver tests passed")


# ============================================================
# Test 5: Boundary conditions
# ============================================================
def test_bcs():
    print("Test 5: Boundary Conditions")
    
    flow_t = flow_inlet(lambda t: 0.2, lambda t: 20.0)
    Q, T = flow_t(0.0)
    assert Q == 0.2, f"Flow BC mismatch: {Q}"
    assert T == 20.0, f"Temp BC mismatch: {T}"
    
    P_t = pressure_outlet(lambda t: 1.0e6)
    assert P_t(0) == 1.0e6
    
    print("  ✅ All BC tests passed")


# ============================================================
# Test 6: Network
# ============================================================
def test_network():
    print("Test 6: Series Network")
    
    pipe1 = Pipe(10000, 0.6, 0.014)
    pipe2 = Pipe(5000, 0.5, 0.012)
    oil = Liquid(name="Oil", rho_ref=860.0, bulk_modulus=1.8e9)
    
    net = SeriesNetwork([
        PipeSegment(pipe1, oil, Nx=30),
        PipeSegment(pipe2, oil, Nx=20),
    ])
    
    assert abs(net.total_length - 15000) < 1
    
    # Steady state
    Pp, Tp, xp = net.steady_state(Q=0.2, P_outlet=0.5e6, T_inlet=50.0)
    assert len(Pp) == 2
    assert len(Tp) == 2
    assert Pp[0][-1] > Pp[1][-1], "Segment 1 outlet > global P_out"
    
    # Transient
    results = net.solve(t_max=10.0, Q_inlet=0.2, P_outlet=0.5e6, T_inlet=50.0)
    assert len(results) == 2
    assert results[0].Nt > 5
    assert results[1].Nt > 5
    
    # Combine
    combined = net.combine_results(results)
    assert combined.x.shape[0] == 52  # 31 + 21
    assert combined.Nt > 5
    
    print("  ✅ All network tests passed")


# ============================================================
# Test 7: Pump
# ============================================================
def test_pump():
    print("Test 7: Pump Model")
    
    pump = Pump(Q_rated=0.3, H_rated=480.0)
    
    # Head should decrease with flow
    H_low = pump.head(0.1)
    H_high = pump.head(0.4)
    assert H_low > H_high, "Head should decrease with flow"
    
    # Affinity laws: double speed → 4x head
    H_double = pump.head(0.3, pump.N_rated * 2)
    H_normal = pump.head(0.3, pump.N_rated)
    assert abs(H_double / H_normal - 4.0) < 0.1, \
        f"Affinity law violation: {H_double/H_normal} vs 4.0"
    
    # Power should be positive
    P = pump.power(0.3)
    assert P > 0, f"Power should be positive: {P}"
    
    # Efficiency should be between 0 and 1
    eff = pump.efficiency(0.3)
    assert 0 <= eff <= 1, f"Efficiency out of range: {eff}"
    
    # NPSH should increase with flow
    npsh_low = pump.npsh_required(0.1)
    npsh_high = pump.npsh_required(0.5)
    assert npsh_high > npsh_low
    
    print("  ✅ All pump tests passed")


# ============================================================
# Test 8: Validation
# ============================================================
def test_validation():
    print("Test 8: Input Validation")
    
    # Bad pipe
    try:
        validate_pipe(Pipe(length=-1, diameter=0.5, wall_thickness=0.012))
        assert False, "Should have raised"
    except ValidationError:
        pass
    
    try:
        validate_pipe(Pipe(length=10000, diameter=0, wall_thickness=0.012))
        assert False, "Should have raised"
    except ValidationError:
        pass
    
    # Bad liquid
    try:
        validate_liquid(Liquid(bulk_modulus=0))
        assert False, "Should have raised"
    except ValidationError:
        pass
    
    valid_pipe = Pipe(10000, 0.5, 0.012)
    valid_liquid = Liquid()
    try:
        validate_pipe(valid_pipe)
        validate_liquid(valid_liquid)
    except ValidationError:
        assert False, "Valid params should not raise"
    
    print("  ✅ All validation tests passed")


# ============================================================
# Test 9: Export
# ============================================================
def test_export():
    print("Test 9: Export")
    
    from sim.solver import SinglePhaseTransientSolver
    from sim.export import export_to_csv
    import tempfile
    
    pipe = Pipe(1000, 0.5, 0.012)
    water = Liquid()
    solver = SinglePhaseTransientSolver(pipe, water, Nx=10)
    
    result = solver.solve(
        t_max=5.0,
        inlet_bc=flow_inlet(lambda t: 0.2, lambda t: 20.0),
        outlet_bc=pressure_outlet(lambda t: 1.0e6),
        mode='A',
        V_initial=1.0, T_initial=20.0,
    )
    
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'test.csv')
        export_to_csv(result, out, pipe)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 100
    
    print("  ✅ All export tests passed")


# ============================================================
# Run all tests
# ============================================================
if __name__ == "__main__":
    tests = [
        test_fluid, test_pipe, test_steady, test_solver,
        test_bcs, test_network, test_pump, test_validation, test_export,
    ]
    
    print("=" * 60)
    print("PipelineSim Test Suite")
    print("=" * 60)
    
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ❌ {test.__name__} FAILED: {e}")
    
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{len(tests)} passed")
    print(f"{'='*60}")
    
    sys.exit(0 if passed == len(tests) else 1)
