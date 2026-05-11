#!/usr/bin/env python3
"""
PipelineSim CLI — Command-line interface for pipeline transient simulation.

Usage:
    pipeline-sim run --config case.json
    pipeline-sim analyze --pipe 15000 --diameter 0.6 --flow 0.25
    pipeline-sim water-hammer --length 10000 --diameter 0.5
    pipeline-sim network run --config network.json
"""

import sys
import os
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cmd_analyze(args):
    """Quick steady-state analysis."""
    from sim.fluid import Liquid
    from sim.pipe import Pipe
    from sim.steady import SteadyStateCalculator
    
    pipe = Pipe(args.length, args.diameter, args.wall_thickness)
    liquid = Liquid()
    calc = SteadyStateCalculator(pipe, liquid)
    
    x, P = calc.pressure_profile(Q=args.flow, P_outlet=args.p_outlet * 1e6)
    _x, T = calc.temperature_profile(Q=args.flow, T_inlet=args.t_inlet)
    
    V = args.flow / pipe.area()
    Re = liquid.rho_ref * V * pipe.diameter / liquid.viscosity_ref
    f = float(pipe.friction_factor(np.array([V]), np.array([Re]))[0])
    
    print(f"\n{'='*50}")
    print(f"  Pipeline Analysis")
    print(f"{'='*50}")
    print(f"  Length:     {pipe.length/1000:.1f} km")
    print(f"  Diameter:   {pipe.diameter*1000:.0f} mm")
    print(f"  Flow:       {args.flow*1000:.0f} L/s")
    print(f"  Velocity:   {V:.2f} m/s")
    print(f"  Re:         {Re:.0f}")
    print(f"  f:          {f:.4f}")
    print(f"  P_outlet:   {args.p_outlet:.3f} MPa")
    print(f"  P_inlet:    {P[0]/1e6:.3f} MPa")
    print(f"  T_inlet:    {args.t_inlet:.1f}°C")
    print(f"  T_outlet:   {T[-1]:.1f}°C")
    print(f"{'='*50}")
    
    return 0


def cmd_run(args):
    """Run a transient simulation from config file."""
    from sim.solver import SinglePhaseTransientSolver, flow_inlet, pressure_inlet, pressure_outlet, flow_outlet
    from sim.fluid import Liquid
    from sim.pipe import Pipe
    from sim.steady import SteadyStateCalculator
    from sim.export import export_to_csv, generate_report
    
    with open(args.config) as f:
        config = json.load(f)
    
    # Parse config
    pipe = Pipe(
        length=config['pipe']['length'],
        diameter=config['pipe']['diameter'],
        wall_thickness=config['pipe'].get('wall_thickness', 0.012),
        roughness=config['pipe'].get('roughness', 4.5e-5),
        young_modulus=config['pipe'].get('young_modulus', 2.07e11),
        heat_transfer_coeff=config['pipe'].get('U', 5.0),
        T_ground=config['pipe'].get('T_ground', 15.0),
    )
    
    fluid_config = config.get('fluid', {})
    liquid = Liquid(
        name=fluid_config.get('name', 'Custom'),
        rho_ref=fluid_config.get('rho_ref', 860.0),
        bulk_modulus=fluid_config.get('bulk_modulus', 1.8e9),
        cp=fluid_config.get('cp', 2000.0),
        viscosity_ref=fluid_config.get('viscosity_ref', 0.05),
        viscosity_temp_coeff=fluid_config.get('viscosity_temp_coeff', 0.02),
    )
    
    # Steady initialization
    mode = config.get('mode', 'A')
    Q_steady = config['boundary'].get('Q_steady', 0.25)
    T_steady = config['boundary'].get('T_steady', 45.0)
    P_steady = config['boundary'].get('P_steady', 0.5e6)
    
    print(f"\n📐 Pipe:  {pipe.length/1000:.1f} km × {pipe.diameter*1000:.0f} mm")
    print(f"💧 Fluid: {liquid.name}")
    print(f"🔧 Mode:  {mode}")
    
    if args.no_steady:
        V0 = config['boundary'].get('V_initial', 0.0)
        P_init = None
        T_init = None
        print("⚠️  Skipping steady-state initialization")
    else:
        calc = SteadyStateCalculator(pipe, liquid)
        V0, P_init, T_init = calc.initialize_transient(
            Q=Q_steady, T_inlet=T_steady, P_outlet=P_steady
        )
        print(f"✅ Steady: V={V0:.3f} m/s, P_in={P_init[0]/1e6:.3f} MPa")
    
    # Solver
    solver = SinglePhaseTransientSolver(
        pipe, liquid,
        Nx=config.get('Nx', 40),
    )
    
    t_max = config.get('t_max', 60.0)
    
    bc = config['boundary']
    if mode == 'A':
        inlet = flow_inlet(
            lambda t: bc.get('Q_func_num', Q_steady),
            lambda t: bc.get('T_func_num', T_steady),
        )
        outlet = pressure_outlet(
            lambda t: bc.get('P_func_num', P_steady),
        )
    else:
        inlet = pressure_inlet(
            lambda t: bc.get('P_func_num', P_init[0] if P_init is not None else 2e6),
            lambda t: bc.get('T_func_num', T_steady),
        )
        outlet = flow_outlet(
            lambda t: bc.get('Q_func_num', Q_steady),
        )
    
    result = solver.solve(
        t_max=t_max,
        inlet_bc=inlet,
        outlet_bc=outlet,
        mode=mode,
        V_initial=V0,
        P_initial=P_init,
        T_initial=T_init,
    )
    
    # Export
    output = args.output or 'output/simulation_result.csv'
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    export_to_csv(result, output, pipe)
    
    report = generate_report(result, pipe, liquid, solver, args.config)
    print(f"\n📊 Results saved to {output}")
    print(report)
    
    return 0


def cmd_waterhammer(args):
    """Classic water hammer test."""
    from sim.fluid import Liquid
    from sim.pipe import Pipe
    from sim.solver import SinglePhaseTransientSolver, flow_inlet, pressure_outlet
    from sim.steady import SteadyStateCalculator
    
    from sim.steady import SteadyStateCalculator
    
    pipe = Pipe(args.length, args.diameter, args.wall_thickness)
    water = Liquid()
    
    V0 = args.flow / pipe.area() if args.flow > 0 else 1.0
    P0 = args.outlet_pressure
    
    # Use steady init for proper initial conditions
    solver = SinglePhaseTransientSolver(pipe, water, Nx=args.Nx)
    calc = SteadyStateCalculator(pipe, water)
    _, P_init, _ = calc.initialize_transient(args.flow, 20.0, P0, solver)
    P0_inlet = P_init[0] if P_init is not None else P0 * 1.5
    
    def inlet_Q(t):
        if t < args.closure_time:
            return args.flow
        return max(0.0, args.flow * (1 - (t - args.closure_time) / 0.05))
    
    result = solver.solve(
        t_max=args.t_max,
        inlet_bc=flow_inlet(inlet_Q, lambda t: 20.0),
        outlet_bc=pressure_outlet(lambda t: P0),
        mode='A',
        V_initial=V0,
        P_initial=P0_inlet,
        T_initial=20.0,
    )
    
    a = pipe.wave_speed(water)
    joukowsky = water.rho_ref * a * V0
    
    # Mode A: flow at inlet, pressure at outlet
    # When inlet flow drops: rarefaction travels down, reflects at outlet
    # Peak pressure at INLET = P0 + Joukowsky (after reflection)
    peak_inlet = np.max(result.P[:, 0]) - P0
    peak_at_valve = np.max(np.max(result.P, axis=0) - P0)
    
    print(f"\n{'='*55}")
    print(f"  Water Hammer Test (Flow Closure at Inlet)")
    print(f"{'='*55}")
    print(f"  Pipe: L={args.length/1000:.0f}km, D={args.diameter*1000:.0f}mm")
    print(f"  Wave speed: a={a:.0f} m/s")
    print(f"  Flow: Q={args.flow*1000:.0f} L/s, V={V0:.2f} m/s")
    print(f"  Joukowsky: ΔP = ρ·a·V = {joukowsky/1e6:.3f} MPa")
    print(f"  Inlet peak:  {np.max(result.P[:, 0])/1e6:.3f} MPa (Δ={peak_inlet/1e6:.3f})")
    print(f"  Global peak: {P0/1e6 + peak_at_valve/1e6:.3f} MPa (Δ={peak_at_valve/1e6:.3f})")
    print(f"  Error vs Joukowsky: {abs(peak_at_valve - joukowsky)/joukowsky*100:.1f}%")
    print(f"{'='*55}")
    
    return 0


def cmd_network(args):
    """Multi-pipe network simulation."""
    from sim.network import PipeSegment, SeriesNetwork
    from sim.fluid import Liquid
    from sim.pipe import Pipe
    
    with open(args.config) as f:
        config = json.load(f)
    
    segments = []
    for seg_config in config['segments']:
        pipe = Pipe(
            length=seg_config['length'],
            diameter=seg_config['diameter'],
            wall_thickness=seg_config.get('wall_thickness', 0.012),
        )
        fluid_cfg = seg_config.get('fluid', config.get('fluid'))
        liquid = Liquid(
            name=fluid_cfg.get('name', 'Custom'),
            rho_ref=fluid_cfg.get('rho_ref', 860.0),
            bulk_modulus=fluid_cfg.get('bulk_modulus', 1.8e9),
        )
        Nx = seg_config.get('Nx', 20)
        segments.append(PipeSegment(pipe, liquid, Nx=Nx, name=seg_config.get('name', '')))
    
    net = SeriesNetwork(segments)
    print(f"\n📐 Network: {len(segments)} segments, {net.total_length/1000:.1f} km total")
    
    results = net.solve(
        t_max=config.get('t_max', 10.0),
        Q_inlet=config.get('Q_inlet', 0.2),
        P_outlet=config.get('P_outlet', 0.5e6),
        T_inlet=config.get('T_inlet', 50.0),
    )
    
    for i, r in enumerate(results):
        print(f"  Segment {i+1}: P={np.min(r.P[:,-1])/1e6:.2f}~{np.max(r.P[:,0])/1e6:.2f} MPa, "
              f"T~{r.T[0,0]:.1f}→{r.T[-1,-1]:.1f}°C")
    
    combined = net.combine_results(results)
    print(f"\n  Combined: {combined.Nt} time steps, {combined.x.shape[0]} nodes")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='PipelineSim — Transient Pipeline Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pipeline-sim analyze --length 15000 --diameter 0.6 --flow 0.25
  pipeline-sim run --config examples/case.json --output results.csv
  pipeline-sim water-hammer --length 10000 --diameter 0.5 --flow 0.2
  pipeline-sim network run --config examples/network.json
        """
    )
    parser.add_argument('--version', action='version', version='PipelineSim 0.3.0')
    
    subparsers = parser.add_subparsers(dest='command', help='Sub-commands')
    
    # analyze
    p_analyze = subparsers.add_parser('analyze', help='Steady-state pipeline analysis')
    p_analyze.add_argument('--length', type=float, default=15000, help='Pipe length (m)')
    p_analyze.add_argument('--diameter', type=float, default=0.6, help='Pipe diameter (m)')
    p_analyze.add_argument('--wall-thickness', type=float, default=0.014, help='Wall thickness (m)')
    p_analyze.add_argument('--flow', type=float, default=0.25, help='Flow rate (m³/s)')
    p_analyze.add_argument('--p-outlet', type=float, default=0.5, help='Outlet pressure (MPa)')
    p_analyze.add_argument('--t-inlet', type=float, default=45.0, help='Inlet temperature (°C)')
    
    # run
    p_run = subparsers.add_parser('run', help='Run transient simulation from config')
    p_run.add_argument('--config', type=str, required=True, help='JSON config file')
    p_run.add_argument('--output', type=str, default=None, help='Output CSV path')
    p_run.add_argument('--no-steady', action='store_true', help='Skip steady-state init')
    
    # water-hammer
    p_wh = subparsers.add_parser('water-hammer', help='Run water hammer test')
    p_wh.add_argument('--length', type=float, default=10000, help='Pipe length (m)')
    p_wh.add_argument('--diameter', type=float, default=0.5, help='Pipe diameter (m)')
    p_wh.add_argument('--wall-thickness', type=float, default=0.012, help='Wall thickness (m)')
    p_wh.add_argument('--flow', type=float, default=0.2, help='Initial flow (m³/s)')
    p_wh.add_argument('--outlet-pressure', type=float, default=1.0e6, help='Outlet pressure (Pa)')
    p_wh.add_argument('--closure-time', type=float, default=5.0, help='Closure start (s)')
    p_wh.add_argument('--t-max', type=float, default=60.0, help='Simulation time (s)')
    p_wh.add_argument('--Nx', type=int, default=50, help='Number of grid cells')
    
    # network
    p_net = subparsers.add_parser('network', help='Multi-pipe network commands')
    net_sub = p_net.add_subparsers(dest='net_command', help='Network sub-command')
    p_net_run = net_sub.add_parser('run', help='Run network simulation')
    p_net_run.add_argument('--config', type=str, required=True, help='Network JSON config')
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        return cmd_analyze(args)
    elif args.command == 'run':
        return cmd_run(args)
    elif args.command == 'water-hammer':
        return cmd_waterhammer(args)
    elif args.command == 'network':
        return cmd_network(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
