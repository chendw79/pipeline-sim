"""
PipelineSim Dashboard
=====================
Interactive Dash web app for pipeline transient simulation visualization.

Usage:  python dashboard/app.py
        # Then open http://0.0.0.0:8050 in browser
"""

import sys, os, json
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback, no_update
import dash

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.solver import SinglePhaseTransientSolver, TransientResult
from sim.solver import flow_inlet, pressure_outlet, pressure_inlet, flow_outlet
from sim.pipe import Pipe
from sim.fluid import Liquid


def step(v0, v1, t0, dur=0.0):
    """Simple step/ramp function for BCs."""
    def f(t):
        if dur <= 0:
            return v1 if t >= t0 else v0
        return v0 + (v1 - v0) * min(1, max(0, (t - t0) / dur))
    return f


# ─── Pipeline visualisation ────────────────────────────────────

def build_pipeline_svg(x_positions, elevation, pressure_mpa=None,
                       p_min=None, p_max=None, time_s=0):
    colours = []
    if pressure_mpa is not None and len(pressure_mpa) == len(x_positions):
        if p_min is None: p_min = np.min(pressure_mpa)
        if p_max is None: p_max = np.max(pressure_mpa)
        p_range = p_max - p_min if p_max > p_min else 1.0
        for p in pressure_mpa:
            frac = (p - p_min) / p_range
            r = int(min(255, max(0, frac * 200 + 55)))
            b = int(min(255, max(0, (1 - frac) * 200 + 55)))
            colours.append(f"rgb({r},80,{b})")
    else:
        colours = ["#4a90d9"] * len(x_positions)

    pipe_trace = go.Scatter(
        x=x_positions / 1000.0, y=elevation,
        mode="lines+markers",
        line=dict(color="#888", width=3),
        marker=dict(size=10, color=colours, line=dict(width=1, color="#555")),
        name="Pipeline",
        hovertemplate="x=%{x:.2f}km<br>z=%{y:.1f}m<br>P=%{customdata[0]:.2f}MPa<br>T=%{customdata[1]:.1f}°C<extra></extra>",
        customdata=np.column_stack(
            [pressure_mpa] if pressure_mpa is not None else [[0]*len(x_positions)]),
    )

    elev_fill = go.Scatter(
        x=np.concatenate([x_positions/1000, x_positions[::-1]/1000]),
        y=np.concatenate([elevation, [np.min(elevation)-20]*len(x_positions)]),
        fill="toself", fillcolor="rgba(100,180,100,0.15)",
        line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", showlegend=False,
    )

    fig = go.Figure(data=[elev_fill, pipe_trace],
                    layout=go.Layout(
                        title=dict(text=f"Pipeline @ t={time_s:.2f}s", font=dict(size=14)),
                        xaxis=dict(title="Distance (km)", showgrid=True, gridcolor="#eee"),
                        yaxis=dict(title="Elevation (m)", showgrid=True, gridcolor="#eee"),
                        hovermode="closest", margin=dict(l=50, r=30, t=40, b=40),
                        height=250, paper_bgcolor="#fafafa", plot_bgcolor="#fafafa",
                        showlegend=False,
                        annotations=[
                            dict(x=x_positions[0]/1000, y=elevation[0], xanchor="right",
                                 text="<b>Inlet</b>", showarrow=True, arrowhead=2, ax=-40, ay=-40),
                            dict(x=x_positions[-1]/1000, y=elevation[-1], xanchor="left",
                                 text="<b>Outlet</b>", showarrow=True, arrowhead=2, ax=40, ay=-40),
                        ]))
    return fig


def build_profiles(result, idx):
    x = result.x / 1000.0
    P = result.P[idx] / 1e6
    T = result.T[idx]
    Q = result.Q[idx] * 1000
    V = result.V[idx]

    profiles = []
    specs = [
        ("Pressure", "MPa", P, "#e74c3c"),
        ("Temperature", "°C", T, "#e67e22"),
        ("Flow Rate", "L/s", Q, "#3498db"),
        ("Velocity", "m/s", V, "#9b59b6"),
    ]
    for name, unit, data, clr in specs:
        f = go.Figure()
        f.add_trace(go.Scatter(x=x, y=data, mode="lines+markers",
                                name=name, line=dict(color=clr, width=2), marker=dict(size=4)))
        f.update_layout(title=name, xaxis_title="Distance (km)",
                        yaxis_title=unit, margin=dict(l=40, r=20, t=30, b=30),
                        height=200, paper_bgcolor="#fafafa", plot_bgcolor="#fafafa")
        profiles.append(f)
    return profiles


def build_phase_diagram(result, idx):
    P = result.P[idx] / 1e6
    Q = result.Q[idx] * 1000
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Q, y=P, mode="lines+markers",
                              marker=dict(size=6, color=list(range(len(P))),
                                          colorscale="Viridis", showscale=True,
                                          colorbar=dict(title="Node")),
                              line=dict(width=1, color="#555"), name="P-Q"))
    fig.update_layout(title="P-Q Phase Diagram", xaxis_title="Flow Q (L/s)",
                      yaxis_title="Pressure (MPa)",
                      margin=dict(l=40, r=20, t=30, b=30), height=220,
                      paper_bgcolor="#fafafa", plot_bgcolor="#fafafa")
    return fig


# ─── Run simulation ───────────────────────────────────────────

def run_sim(mode="A", t_max=10.0, Nx=40, q_in=0.25, p_out=2e6, t_in=20.0,
            length=15000, diameter=0.6):
    liq = Liquid("Crude Oil", rho_ref=860, viscosity_ref=0.05,
                 bulk_modulus=1.8e9, cp=2000, thermal_expansion=0.0008,
                 visc_T_coeff=0.02)
    pipe = Pipe(length, diameter, 0.008, 3e-5, 200e9,
                elevation_start=0.0, elevation_end=2.0,
                heat_transfer_coeff=10.0, T_ground=10.0)
    solver = SinglePhaseTransientSolver(pipe, liq, Nx=Nx)
    wave = pipe.wave_speed(liq)

    if mode == "A":
        inlet = flow_inlet(step(q_in, q_in, 0), step(t_in, t_in, 0))
        outlet = pressure_outlet(step(p_out, p_out, 0))
    else:
        inlet = pressure_inlet(step(2.5e6, 2.5e6, 0), step(t_in, t_in, 0))
        outlet = flow_outlet(step(q_in, q_in, 0))

    result = solver.solve(t_max, inlet, outlet, mode=mode)
    return result, pipe, wave


# ─── Dash App ──────────────────────────────────────────────────

app = Dash(__name__, title="PipelineSim Dashboard")



# ─── Layout ────────────────────────────────────────────────────

app.layout = html.Div([
    dcc.Store(id="store-result"),
    dcc.Store(id="store-wave", data=1128),

    # Header
    html.Div([
        html.H1("🛸 PipelineSim", style={"display":"inline-block","margin":"0 20px","fontSize":24}),
        html.Span("Transient Simulation Dashboard", style={"color":"#bdc3c7","fontSize":14}),
    ], style={"background":"#2c3e50","color":"white","padding":"10px 20px",
              "display":"flex","alignItems":"center","gap":10}),

    # Controls
    html.Div([
        html.Div([
            html.Label("Mode:", style={"fontWeight":"bold","marginRight":8}),
            dcc.RadioItems(id="radio-mode", options=[
                {"label":" A: Inlet Q+T / Outlet P","value":"A"},
                {"label":" B: Inlet P+T / Outlet Q","value":"B"},
            ], value="A", inline=True, inputStyle={"marginRight":4}),
        ], style={"display":"flex","alignItems":"center","gap":10}),
        html.Div([
            html.Button("▶ Run", id="btn-run",
                        style={"background":"#27ae60","color":"white","border":"none",
                               "padding":"8px 24px","borderRadius":4,"cursor":"pointer","fontWeight":"bold"}),
            html.Button("⚙ Config", id="btn-config",
                        style={"background":"#7f8c8d","color":"white","border":"none",
                               "padding":"8px 16px","borderRadius":4,"cursor":"pointer"}),
        ], style={"display":"flex","gap":8}),
    ], style={"background":"#ecf0f1","padding":"10px 20px",
              "display":"flex","justifyContent":"space-between","alignItems":"center",
              "borderBottom":"1px solid #bdc3c7"}),

    # Pipeline + Metrics row
    html.Div([
        html.Div([dcc.Graph(id="graph-pipeline", style={"height":260})], style={"flex":2}),
        html.Div(id="metrics-panel", children=[
            html.Div([html.Span("Max P",className="metric-label"),
                      html.Span("—",id="metric-pmax",className="metric-value")],className="metric-card"),
            html.Div([html.Span("Min P",className="metric-label"),
                      html.Span("—",id="metric-pmin",className="metric-value")],className="metric-card"),
            html.Div([html.Span("ΔP",className="metric-label"),
                      html.Span("—",id="metric-dp",className="metric-value")],className="metric-card"),
            html.Div([html.Span("T avg",className="metric-label"),
                      html.Span("—",id="metric-tavg",className="metric-value")],className="metric-card"),
            html.Div([html.Span("Q in",className="metric-label"),
                      html.Span("—",id="metric-qin",className="metric-value")],className="metric-card"),
            html.Div([html.Span("Wave",className="metric-label"),
                      html.Span("—",id="metric-wave",className="metric-value")],className="metric-card"),
        ], style={"flex":1,"display":"flex","flexWrap":"wrap","gap":6,"padding":"8px"}),
    ], style={"display":"flex","gap":0,"flexWrap":"wrap"}),

    # Profile plots 2x2
    html.Div([
        html.Div([dcc.Graph(id="graph-press", style={"height":220})], style={"flex":1}),
        html.Div([dcc.Graph(id="graph-temp", style={"height":220})], style={"flex":1}),
    ], style={"display":"flex","gap":4}),
    html.Div([
        html.Div([dcc.Graph(id="graph-flow", style={"height":220})], style={"flex":1}),
        html.Div([dcc.Graph(id="graph-vel", style={"height":220})], style={"flex":1}),
    ], style={"display":"flex","gap":4}),

    # Phase diagram + time controls
    html.Div([
        html.Div([dcc.Graph(id="graph-phase", style={"height":220})], style={"flex":1}),
        html.Div([
            html.Label("⏱ Time", style={"fontWeight":"bold"}),
            dcc.Slider(id="slider-time", min=0, max=100, step=1, value=0,
                       tooltip={"placement":"bottom","always_visible":True}, marks=None),
            html.Div([
                dcc.Checklist(id="chk-play", options=[{"label":" Auto-Play","value":"play"}],
                              value=[], inline=True, style={"marginTop":8}),
                html.Span(id="time-label", style={"marginLeft":16,"color":"#7f8c8d"}),
            ], style={"display":"flex","alignItems":"center"}),
            dcc.Interval(id="interval-anim", interval=150, n_intervals=0, disabled=True),
        ], style={"flex":1,"padding":"10px 20px"}),
    ], style={"display":"flex","gap":4}),

    # Config modal
    html.Div(id="config-modal", children=[
        html.Div([
            html.H3("Configuration", style={"margin":"0 0 12px 0"}),
            html.Label("Pipe Length (m)"),
            dcc.Input(id="cfg-length", type="number", value=15000, style={"width":"100%"}),
            html.Label("Diameter (mm)"),
            dcc.Input(id="cfg-diam", type="number", value=600, style={"width":"100%"}),
            html.Label("Flow (m³/s)"),
            dcc.Input(id="cfg-qin", type="number", value=0.25, step=0.01, style={"width":"100%"}),
            html.Label("Outlet P (MPa)"),
            dcc.Input(id="cfg-pout", type="number", value=2.0, step=0.1, style={"width":"100%"}),
            html.Label("Sim Time (s)"),
            dcc.Input(id="cfg-tmax", type="number", value=10.0, step=1, style={"width":"100%"}),
            html.Br(),
            html.Button("Close", id="btn-close-config",
                        style={"background":"#e74c3c","color":"white","border":"none",
                               "padding":"8px 24px","borderRadius":4,"cursor":"pointer"}),
        ], style={"background":"white","padding":20,"maxWidth":360,"margin":"auto",
                  "borderRadius":8,"boxShadow":"0 4px 20px rgba(0,0,0,0.3)"}),
    ], style={"display":"none"}),
])


# ─── Callbacks ─────────────────────────────────────────────────

@callback(
    [Output("store-result","data"),
     Output("store-wave","data"),
     Output("graph-pipeline","figure"),
     Output("graph-press","figure"),
     Output("graph-temp","figure"),
     Output("graph-flow","figure"),
     Output("graph-vel","figure"),
     Output("graph-phase","figure"),
     Output("slider-time","max"),
     Output("slider-time","value"),
     Output("metric-pmax","children"),
     Output("metric-pmin","children"),
     Output("metric-dp","children"),
     Output("metric-tavg","children"),
     Output("metric-qin","children"),
     Output("metric-wave","children")],
    Input("btn-run","n_clicks"),
    State("radio-mode","value"),
    prevent_initial_call=True,
)
def on_run(n, mode):
    result, pipe, wave = run_sim(mode=mode, t_max=10, Nx=40)
    data = dict(t=result.t.tolist(), x=result.x.tolist(),
                P=result.P.tolist(), T=result.T.tolist(),
                V=result.V.tolist(), Q=result.Q.tolist())

    z = pipe.elevation(result.x)
    P0 = result.P[0]/1e6
    fig_pipe = build_pipeline_svg(result.x, z, P0, np.min(P0), np.max(P0), 0)

    fig_P, fig_T, fig_Q, fig_V = build_profiles(result, 0)
    fig_ph = build_phase_diagram(result, 0)

    Pend = result.P[-1]/1e6
    return (data, wave, fig_pipe, fig_P, fig_T, fig_Q, fig_V, fig_ph,
            len(result.t)-1, 0,
            f"{np.max(Pend):.2f} MPa", f"{np.min(Pend):.2f} MPa",
            f"{(np.max(Pend)-np.min(Pend))*1000:.0f} kPa",
            f"{np.mean(result.T[-1]):.1f} °C",
            f"{result.Q[-1,0]*1000:.1f} L/s",
            f"{wave:.0f} m/s")


@callback(
    [Output("graph-pipeline","figure",allow_duplicate=True),
     Output("graph-press","figure",allow_duplicate=True),
     Output("graph-temp","figure",allow_duplicate=True),
     Output("graph-flow","figure",allow_duplicate=True),
     Output("graph-vel","figure",allow_duplicate=True),
     Output("graph-phase","figure",allow_duplicate=True),
     Output("metric-pmax","children",allow_duplicate=True),
     Output("metric-pmin","children",allow_duplicate=True),
     Output("metric-dp","children",allow_duplicate=True),
     Output("metric-tavg","children",allow_duplicate=True),
     Output("metric-qin","children",allow_duplicate=True),
     Output("time-label","children")],
    Input("slider-time","value"),
    State("store-result","data"),
    prevent_initial_call=True,
)
def on_slider(idx, data):
    if data is None or idx is None:
        return [no_update]*12
    i = min(idx, len(data["t"])-1)
    class R: pass
    r = R()
    r.t = np.array(data["t"])
    r.x = np.array(data["x"])
    r.P = np.array(data["P"])
    r.T = np.array(data["T"])
    r.V = np.array(data["V"])
    r.Q = np.array(data["Q"])

    L = r.x[-1]
    z = 2 * np.sin(np.pi * r.x / L)
    Pi = r.P[i]/1e6
    fig_pipe = build_pipeline_svg(r.x, z, Pi, np.min(r.P), np.max(r.P), r.t[i])
    fig_P, fig_T, fig_Q, fig_V = build_profiles(r, i)
    fig_ph = build_phase_diagram(r, i)

    Pcur = r.P[i]/1e6
    return (fig_pipe, fig_P, fig_T, fig_Q, fig_V, fig_ph,
            f"{np.max(Pcur):.2f} MPa", f"{np.min(Pcur):.2f} MPa",
            f"{(np.max(Pcur)-np.min(Pcur))*1000:.0f} kPa",
            f"{np.mean(r.T[i]):.1f} °C",
            f"{r.Q[i,0]*1000:.1f} L/s",
            f"t = {r.t[i]:.2f}s")


@callback(
    [Output("slider-time","value",allow_duplicate=True),
     Output("interval-anim","disabled"),
     Output("interval-anim","n_intervals")],
    [Input("chk-play","value"), Input("interval-anim","n_intervals")],
    State("store-result","data"),
    prevent_initial_call=True,
)
def anim(chk, n, data):
    ctx = dash.callback_context
    trig = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
    if data is None or len(data["t"]) < 2:
        return 0, True, 0
    max_i = len(data["t"])-1
    if "chk-play" in trig:
        return 0, "play" not in chk, 0
    if "play" not in chk:
        return no_update, True, 0
    return (n % (max_i+1), False, n+1)


@callback(
    Output("config-modal","style"),
    Input("btn-config","n_clicks"),
    Input("btn-close-config","n_clicks"),
    prevent_initial_call=True,
)
def toggle_config(n1, n2):
    ctx = dash.callback_context
    if not ctx.triggered:
        return {"display":"none"}
    if "btn-config" in ctx.triggered[0]["prop_id"]:
        return {"position":"fixed","top":0,"left":0,"right":0,"bottom":0,
                "background":"rgba(0,0,0,0.5)","zIndex":1000,
                "display":"flex","alignItems":"center","justifyContent":"center"}
    return {"display":"none"}


# ─── CSS ───────────────────────────────────────────────────────

app.index_string = app.index_string.replace(
    "</head>",
    """<style>
*{box-sizing:border-box}
body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,
     Helvetica,Arial,sans-serif;background:#f5f6fa}
.metric-card{background:white;border-radius:6px;padding:10px 14px;min-width:100px;
             flex:1;box-shadow:0 1px 3px rgba(0,0,0,.08);text-align:center}
.metric-label{display:block;font-size:11px;color:#7f8c8d;text-transform:uppercase;
              letter-spacing:.5px;margin-bottom:4px}
.metric-value{display:block;font-size:18px;font-weight:bold;color:#2c3e50}
</style></head>"""
)


# ─── Entry point ───────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser(description="PipelineSim Dashboard")
    p.add_argument("--port", type=int, default=8050)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
