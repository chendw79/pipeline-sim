#!/usr/bin/env python3
"""
PipelineSim Interactive Dashboard Generator
Generates a single, self-contained HTML dashboard using Plotly.
"""

import os, sys, json, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sim.fluid import Liquid
from sim.pipe import Pipe
from sim.solver import flow_inlet, pressure_outlet, flow_outlet, pressure_inlet
from sim.solver_advanced import (
    compare_solvers, ImplicitFDMSolver, MacCormackSolver, FiniteVolumeSolver
)
from sim.solver import SinglePhaseTransientSolver
from sim.steady import SteadyStateCalculator
import plotly.graph_objects as go
import plotly.io as pio


# ============================================================
# DASHBOARD CONFIG
# ============================================================
OUTPUT = os.path.join(os.path.dirname(__file__), '..', 'output', 'pipeline_dashboard.html')
DEMO_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'demo')
os.makedirs(DEMO_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

DARK_THEME = dict(
    plot_bgcolor='#1a1a2e', paper_bgcolor='#16213e',
    font=dict(color='#e0e0e0', size=11),
    xaxis=dict(gridcolor='#2a2a4e', zerolinecolor='#3a3a5e'),
    yaxis=dict(gridcolor='#2a2a4e', zerolinecolor='#3a3a5e'),
    margin=dict(l=60, r=30, t=40, b=60),
    hovermode='x unified',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
)

PALETTE = ['#00d4ff', '#ff6b6b', '#ffd93d', '#6bcb77', '#a66cff', '#ff8e53']


def run_solver_comparison():
    """Run all 4 solvers and return results + performance data."""
    pipe = Pipe(10000, 0.5, 0.012)
    liquid = Liquid()
    
    def inlet_Q(t):
        if t < 2.0: return 0.2
        elif t < 4.0: return 0.2*(1-(t-2.0)/2.0)
        else: return 0.0
    
    perf = {}
    t0 = time.time()
    results = compare_solvers(pipe, liquid,
        inlet_bc=flow_inlet(inlet_Q, lambda t: 20.0),
        outlet_bc=pressure_outlet(lambda t: 1.0e6),
        mode='A', Nx=30, t_max=30.0, Q0=0.2, T0=20.0, P_out=1.0e6)
    elapsed = time.time() - t0
    
    for name, r in results.items():
        # Find peak time
        peak_idx = np.argmax(r.P.max(axis=1))
        perf[name] = dict(
            Pmax=r.P.max()/1e6, t_peak=r.t[peak_idx],
            Nt=r.Nt, Nx=r.Nx,
        )
    
    return pipe, liquid, results, perf, elapsed


def fig_solver_wave_comparison(results, pipe):
    """4-line pressure wave at t=0,5,10,15s."""
    figs = []
    for t_show in [0, 5, 10, 15]:
        fig = go.Figure()
        for name, r in results.items():
            idx = np.argmin(np.abs(r.t - t_show))
            fig.add_trace(go.Scatter(
                x=r.x/1000, y=r.P[idx]/1e6,
                mode='lines', name=name,
                line=dict(width=2),
            ))
        fig.update_layout(
            title=f'Pressure Profile at t={t_show}s',
            xaxis_title='Distance (km)', yaxis_title='Pressure (MPa)',
            **DARK_THEME, height=300, showlegend=(t_show == 0),
        )
        figs.append(fig)
    return figs


def fig_inlet_pressure(results):
    """Inlet pressure over time, all solvers."""
    fig = go.Figure()
    for name, r in results.items():
        fig.add_trace(go.Scatter(
            x=r.t, y=r.P[:, 0]/1e6,
            mode='lines', name=name,
            line=dict(width=1.5),
        ))
    fig.update_layout(
        title='Inlet Pressure Time History',
        xaxis_title='Time (s)', yaxis_title='Pressure (MPa)',
        **DARK_THEME, height=350,
    )
    return fig


def fig_outlet_flow(results):
    """Outlet flow rate over time."""
    fig = go.Figure()
    for name, r in results.items():
        fig.add_trace(go.Scatter(
            x=r.t, y=r.Q[:, -1],
            mode='lines', name=name,
            line=dict(width=1.5),
        ))
    fig.update_layout(
        title='Outlet Flow Rate Time History',
        xaxis_title='Time (s)', yaxis_title='Flow Rate (m³/s)',
        **DARK_THEME, height=350,
    )
    return fig


def fig_pressure_animation(results):
    """Pressure wave propagation animation (slider)."""
    fig = go.Figure()
    r_moc = results.get('MOC', list(results.values())[0])
    
    # Add traces for all solvers at initial time
    for name, r in results.items():
        fig.add_trace(go.Scatter(
            x=r.x/1000, y=r.P[0]/1e6,
            mode='lines', name=name,
            line=dict(width=2),
        ))
    
    # Create slider steps
    n_steps = min(50, r_moc.Nt)
    step_indices = np.linspace(0, r_moc.Nt-1, n_steps, dtype=int)
    
    steps = []
    for idx in step_indices:
        t_val = r_moc.t[idx]
        step = dict(
            method='update',
            args=[{
                'y': [r.P[min(idx, r.Nt-1)]/1e6 for r in results.values()]
            }, {'title': f'Pressure Wave at t={t_val:.1f}s'}],
            label=f'{t_val:.0f}s',
        )
        steps.append(step)
    
    sliders = [dict(
        active=0, currentvalue=dict(prefix='Time: ', font=dict(size=12)),
        pad=dict(t=20), len=0.9, x=0.05,
        steps=steps,
    )]
    
    fig.update_layout(
        title='Pressure Wave Propagation Animation',
        xaxis_title='Distance (km)', yaxis_title='Pressure (MPa)',
        sliders=sliders, **DARK_THEME, height=400,
    )
    return fig


def fig_temperature_animation(results):
    """Temperature profile animation."""
    r_moc = results.get('MOC', list(results.values())[0])
    fig = go.Figure()
    
    for name, r in results.items():
        fig.add_trace(go.Scatter(
            x=r.x/1000, y=r.T[0],
            mode='lines', name=name,
            line=dict(width=2),
        ))
    
    n_steps = min(50, r_moc.Nt)
    step_indices = np.linspace(0, r_moc.Nt-1, n_steps, dtype=int)
    
    steps = []
    for idx in step_indices:
        t_val = r_moc.t[idx]
        step = dict(
            method='update',
            args=[{
                'y': [r.T[min(idx, r.Nt-1)] for r in results.values()]
            }],
            label=f'{t_val:.0f}s',
        )
        steps.append(step)
    
    sliders = [dict(
        active=0, currentvalue=dict(prefix='Time: ', font=dict(size=12)),
        pad=dict(t=20), len=0.9, x=0.05, steps=steps,
    )]
    
    fig.update_layout(
        title='Temperature Profile Animation',
        xaxis_title='Distance (km)', yaxis_title='Temperature (°C)',
        sliders=sliders, **DARK_THEME, height=400,
    )
    return fig


def run_demo_case(pipe, liquid, inlet_bc, outlet_bc, mode, case_name, Nx=30, t_max=30.0):
    """Run a demo case with all compatible solvers."""
    from sim.solver_advanced import compare_solvers
    Q0 = liquid.rho_ref  # placeholder
    T0 = 20.0
    P_out = 1e6
    
    # Determine Q0 and P_out from BCs
    if mode == 'A':
        q_test, _ = inlet_bc(0)
        Q0 = q_test
        P_out = outlet_bc(0) if hasattr(outlet_bc, '__call__') else 1e6
    else:
        p_test, _ = inlet_bc(0)
        P_out = p_test
        Q0 = 0.1
    
    results = compare_solvers(pipe, liquid,
        inlet_bc=inlet_bc, outlet_bc=outlet_bc,
        mode=mode, Nx=Nx, t_max=t_max,
        Q0=Q0, T0=T0, P_out=P_out,
        compare_methods=['MOC', 'MacCormack', 'FVM', 'IFDM'],
    )
    return results


def fig_demo_case(results, case_name):
    """Create figures for a demo case."""
    fig = go.Figure()
    for name, r in results.items():
        fig.add_trace(go.Scatter(
            x=r.t, y=r.P[:, 0]/1e6,
            mode='lines', name=name,
            line=dict(width=2),
        ))
    
    peak_times = {name: r.t[np.argmax(r.P.max(axis=1))] for name, r in results.items()}
    peak_press = {name: r.P.max()/1e6 for name, r in results.items()}
    
    fig.update_layout(
        title=f'{case_name}',
        xaxis_title='Time (s)', yaxis_title='Inlet Pressure (MPa)',
        **DARK_THEME, height=350,
    )
    
    # Summary table
    summary = f"<b>{case_name}</b><br>"
    summary += "<table style='width:100%;color:#e0e0e0;font-size:12px;border-collapse:collapse'>"
    summary += "<tr><th>Solver</th><th>Pmax (MPa)</th><th>Peak Time (s)</th></tr>"
    for name in ['MOC', 'MacCormack', 'FVM', 'IFDM']:
        if name in results:
            summary += f"<tr><td>{name}</td><td>{peak_press[name]:.3f}</td><td>{peak_times[name]:.1f}</td></tr>"
    summary += "</table>"
    
    return fig, summary


def fig_pipe_profile(pipe, liquid, results=None):
    """Pipeline elevation + pressure profile."""
    x_km = np.linspace(0, pipe.length/1000, 100)
    x_m = x_km * 1000
    z = pipe.elevation(x_m)
    
    theme = DARK_THEME.copy()
    theme['hovermode'] = 'closest'
    
    fig = go.Figure()
    
    # Elevation area
    fig.add_trace(go.Scatter(
        x=x_km, y=z, fill='tozeroy',
        mode='lines', name='Elevation',
        line=dict(color='#6bcb77', width=2),
        fillcolor='rgba(107, 203, 119, 0.2)',
        hovertemplate='x=%{x:.1f}km<br>z=%{y:.1f}m<extra>Elevation</extra>',
    ))
    
    # Pressure overlay (from MOC or latest result)
    if results and 'MOC' in results:
        r = results['MOC']
        final_idx = -1
        p_overlay = r.P[final_idx] / 1e6
        
        # Normalize for visual overlay
        p_norm = p_overlay / max(p_overlay) * max(z) * 0.5 if max(z) > 0 else p_overlay * 5
        fig.add_trace(go.Scatter(
            x=r.x/1000, y=p_norm,
            mode='lines+markers', name=f'Pressure (t={r.t[final_idx]:.1f}s)',
            line=dict(color='#00d4ff', width=2, dash='dash'),
            marker=dict(size=4, color=p_overlay, colorscale='RdYlBu_r',
                        cmin=0.5, cmax=2.5, showscale=True,
                        colorbar=dict(title='MPa', x=1.02)),
            hovertemplate='x=%{x:.2f}km<br>P=%{customdata:.3f}MPa<extra></extra>',
            customdata=p_overlay,
        ))
    
    fig.update_layout(
        title='Pipeline Profile + Pressure Overlay',
        xaxis_title='Distance (km)', yaxis_title='Elevation (m)',
        **theme, height=400,
    )
    return fig


def generate_html(all_results, perf, elapsed, pipe, demo_results=None):
    """Assemble all figures into a single HTML with tab navigation."""
    r = all_results
    
    tab1_content = '<div id="tab1" class="tab-content active">'
    # Solver comparison panel
    tab1_content += '<h2>🧪 Solver Comparison</h2>'
    tab1_content += '<div class="grid-2col">'
    
    # Pressure wave snapshots
    for fig in fig_solver_wave_comparison(r, pipe):
        tab1_content += f'<div>{pio.to_html(fig, include_plotlyjs=False, full_html=False)}</div>'
    
    tab1_content += '</div>'
    
    # Inlet pressure and outlet flow side by side
    tab1_content += '<div class="grid-2col">'
    tab1_content += f'<div>{pio.to_html(fig_inlet_pressure(r), include_plotlyjs=False, full_html=False)}</div>'
    tab1_content += f'<div>{pio.to_html(fig_outlet_flow(r), include_plotlyjs=False, full_html=False)}</div>'
    tab1_content += '</div>'
    
    # Performance table
    tab1_content += '<h3>📊 Solver Performance Summary</h3>'
    tab1_content += '<div style="overflow-x:auto"><table class="data-table">'
    tab1_content += '<tr><th>Solver</th><th>Pmax (MPa)</th><th>Peak Time (s)</th><th>Time Steps</th></tr>'
    for name, p in perf.items():
        color = '#4ade80' if 1.9 <= p['Pmax'] <= 2.2 else '#f87171'
        tab1_content += f'<tr><td>{name}</td><td style="color:{color}">{p["Pmax"]:.3f}</td><td>{p["t_peak"]:.1f}</td><td>{p["Nt"]}</td></tr>'
    tab1_content += '</table></div>'
    tab1_content += '</div>'
    
    tab2_content = '<div id="tab2" class="tab-content">'
    tab2_content += '<h2>🎬 Animation Panel</h2>'
    tab2_content += f'<div>{pio.to_html(fig_pressure_animation(r), include_plotlyjs=False, full_html=False)}</div>'
    tab2_content += f'<div>{pio.to_html(fig_temperature_animation(r), include_plotlyjs=False, full_html=False)}</div>'
    tab2_content += '</div>'
    
    tab3_content = '<div id="tab3" class="tab-content">'
    tab3_content += '<h2>🔬 Demo Cases</h2>'
    if demo_results:
        for case_name, r_dict in demo_results.items():
            fig, summary = fig_demo_case(r_dict, case_name)
            tab3_content += f'<div>{pio.to_html(fig, include_plotlyjs=False, full_html=False)}</div>'
            tab3_content += f'<div>{summary}</div>'
    tab3_content += '</div>'
    
    tab4_content = '<div id="tab4" class="tab-content">'
    tab4_content += '<h2>📐 Pipeline Profile</h2>'
    tab4_content += f'<div>{pio.to_html(fig_pipe_profile(pipe, Liquid(), r), include_plotlyjs=False, full_html=False)}</div>'
    tab4_content += '<p>Hover over pipeline nodes to see elevation, pressure, temperature data.</p>'
    tab4_content += '</div>'
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PipelineSim Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: #0f0f23; color: #e0e0e0; font-family: 'Segoe UI', Arial, sans-serif; }}
.header {{ background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 20px 30px; border-bottom: 2px solid #00d4ff; }}
.header h1 {{ font-size: 24px; color: #00d4ff; }}
.header p {{ font-size: 13px; color: #888; margin-top: 4px; }}
.tabs {{ display: flex; gap: 0; background: #1a1a2e; border-bottom: 1px solid #2a2a4e; position: sticky; top: 0; z-index: 100; }}
.tab-btn {{ padding: 12px 24px; background: transparent; color: #888; border: none; cursor: pointer; font-size: 14px; font-weight: 600; transition: all 0.2s; }}
.tab-btn:hover {{ color: #00d4ff; background: rgba(0,212,255,0.1); }}
.tab-btn.active {{ color: #00d4ff; border-bottom: 3px solid #00d4ff; }}
.tab-content {{ display: none; padding: 20px; }}
.tab-content.active {{ display: block; }}
.grid-2col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 12px 0; }}
@media (max-width: 900px) {{ .grid-2col {{ grid-template-columns: 1fr; }} }}
.data-table {{ width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 13px; }}
.data-table th {{ background: #1a1a2e; color: #00d4ff; padding: 10px 14px; text-align: left; border-bottom: 2px solid #2a2a4e; }}
.data-table td {{ padding: 8px 14px; border-bottom: 1px solid #1e1e3a; }}
.data-table tr:hover {{ background: rgba(0,212,255,0.05); }}
.footer {{ padding: 16px 30px; background: #1a1a2e; border-top: 1px solid #2a2a4e; font-size: 12px; color: #666; }}
h2 {{ color: #00d4ff; font-size: 18px; margin: 12px 0; }}
h3 {{ color: #aad; font-size: 15px; margin: 16px 0 8px; }}
</style>
</head>
<body>
<div class="header">
  <h1>🛸 PipelineSim Interactive Dashboard</h1>
  <p>Transient Pipeline Analyzer | Wave Speed: {pipe.wave_speed(Liquid()):.0f} m/s | Pipe: {pipe.length/1000:.0f}km × {pipe.diameter*1000:.0f}mm</p>
</div>
<div class="tabs">
  <button class="tab-btn active" onclick="switchTab('tab1')">🧪 Solver Comparison</button>
  <button class="tab-btn" onclick="switchTab('tab2')">🎬 Animation</button>
  <button class="tab-btn" onclick="switchTab('tab3')">🔬 Demo Cases</button>
  <button class="tab-btn" onclick="switchTab('tab4')">📐 Pipeline Profile</button>
</div>
<div class="content">
{tab1_content}
{tab2_content}
{tab3_content}
{tab4_content}
</div>
<div class="footer">
  <span>Computation: {elapsed:.2f}s | Nx=30 | Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} | PipelineSim v2.0</span>
</div>
<script>
function switchTab(id) {{
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  document.querySelector(`button[onclick="switchTab('${{id}}')"]`).classList.add('active');
  // Resize plotly charts after tab switch
  setTimeout(() => {{ window.dispatchEvent(new Event('resize')); }}, 100);
}}
</script>
</body>
</html>'''
    
    with open(OUTPUT, 'w') as f:
        f.write(html)
    print(f'✅ Dashboard saved: {OUTPUT}')


if __name__ == '__main__':
    print('=' * 60)
    print('  Generating Pipeline Dashboard...')
    print('=' * 60)
    
    pipe, liquid, results, perf, elapsed = run_solver_comparison()
    
    # Demo cases
    demo_results = {}
    
    # Case 1: Valve closure (water hammer)
    print('\n  Running Demo Case 1: Valve Closure (10km, fast)...')
    pipe1 = Pipe(10000, 0.5, 0.012)
    liquid1 = Liquid()
    def inlet_Q_vc(t):
        if t < 1.0: return 0.2
        elif t < 3.0: return 0.2*(1-(t-1.0)/2.0)
        else: return 0.0
    r1 = compare_solvers(pipe1, liquid1,
        inlet_bc=flow_inlet(inlet_Q_vc, lambda t: 20.0),
        outlet_bc=pressure_outlet(lambda t: 1.0e6),
        mode='A', Nx=30, t_max=30.0, Q0=0.2, T0=20.0, P_out=1.0e6,
        compare_methods=['MOC', 'MacCormack', 'FVM', 'IFDM'])
    demo_results['Case 1: Valve Closure (Water Hammer)'] = r1
    
    print('  Running Demo Case 2: Pump Trip (20km, pressure drop)...')
    pipe2 = Pipe(20000, 0.6, 0.015)
    liquid2 = Liquid()
    def inlet_P_pt(t):
        if t < 5.0: return 4.0e6
        elif t < 15.0: return 4.0e6 * (1 - (t-5.0)/10.0)
        else: return 2.0e6
    def outlet_Q_pt(t):
        return 0.3
    r2 = compare_solvers(pipe2, liquid2,
        inlet_bc=pressure_inlet(inlet_P_pt, lambda t: 20.0),
        outlet_bc=flow_outlet(outlet_Q_pt),
        mode='B', Nx=40, t_max=60.0, Q0=0.3, T0=20.0, P_out=4e6,
        compare_methods=['MOC', 'MacCormack', 'FVM', 'IFDM'])
    demo_results['Case 2: Pump Trip (Pressure Drop, 20km)'] = r2
    
    generate_html(results, perf, elapsed, pipe, demo_results)
