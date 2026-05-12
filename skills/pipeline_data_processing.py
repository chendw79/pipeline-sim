"""
PipelineSim 数据处理与可视化模块
================================
专业级管道瞬态数据处理技能。

核心能力：
1. 数据面板 — 实时KPI指标卡（最大压力/最低压力/流量/温度）
2. 沿程分布 — 沿线压力/温度/流速/密度剖面
3. 时程曲线 — 任意位置随时间变化趋势 + 多点对比

输出：Plotly交互式HTML / Matplotlib工程报告图
"""

import os, json, sys
from typing import List, Dict, Optional, Tuple
import numpy as np

# ─── 数据面板 (KPI Cards) ─────────────────────────────────

class PipelineDataPanel:
    """管道数据面板 — KPI指标卡生成"""
    
    @staticmethod
    def compute_kpis(result, render_mode='dict'):
        """
        从瞬态结果计算KPI指标
        
        Args:
            result: TransientResult对象 (含 x, t, P, T, rho, v 等)
            render_mode: 'dict' / 'html' / 'plotly'
        
        Returns:
            包含所有关键指标的字典
        """
        P = getattr(result, 'P', None)   # (nt, nx)
        T = getattr(result, 'T', None)
        rho = getattr(result, 'rho', None)
        v = getattr(result, 'v', None)
        t_arr = getattr(result, 't', None)
        x_arr = getattr(result, 'x', None)
        
        if P is None:
            return {'error': '无效的结果对象'}
        
        kpis = {
            'max_pressure_mpa': float(np.max(P)) / 1e6,
            'max_pressure_location_km': float(x_arr[np.argmax(P[-1])] / 1000) if t_arr is not None else 0,
            'max_pressure_time_s': float(t_arr[np.unravel_index(np.argmax(P), P.shape)[0]]) if t_arr is not None else 0,
            'min_pressure_mpa': float(np.min(P)) / 1e6,
            'pressure_spread_mpa': float((np.max(P) - np.min(P)) / 1e6),
            
            'inlet_pressure_mpa': float(P[-1, 0] / 1e6) if len(P.shape) > 1 else float(P[-1] / 1e6),
            'outlet_pressure_mpa': float(P[-1, -1] / 1e6) if len(P.shape) > 1 else float(P[-1] / 1e6),
            'pressure_drop_mpa': float((P[-1, 0] - P[-1, -1]) / 1e6) if len(P.shape) > 1 else 0,
        }
        
        # 温度
        if T is not None:
            kpis['max_temp_c'] = float(np.max(T)) - 273.15
            kpis['min_temp_c'] = float(np.min(T)) - 273.15
            kpis['inlet_temp_c'] = float(T[-1, 0] - 273.15) if len(T.shape) > 1 else float(T[-1] - 273.15)
        
        # 流速
        if v is not None:
            kpis['max_velocity_ms'] = float(np.max(v))
            kpis['min_velocity_ms'] = float(np.min(v))
        
        # 密度
        if rho is not None:
            kpis['max_density_kgm3'] = float(np.max(rho))
            kpis['min_density_kgm3'] = float(np.min(rho))
        
        # 总时长
        if t_arr is not None:
            kpis['total_time_s'] = float(t_arr[-1])
            kpis['total_time_min'] = float(t_arr[-1] / 60)
            kpis['num_steps'] = len(t_arr)
        
        # 管线信息
        if x_arr is not None:
            kpis['pipeline_length_km'] = float(x_arr[-1] / 1000)
            kpis['num_nodes'] = len(x_arr)
        
        return kpis
    
    @staticmethod
    def kpis_to_html(kpis: dict, title: str = "管道瞬态分析 KPI") -> str:
        """KPI → HTML面板"""
        cards = []
        color_pairs = [
            ('#ff6b6b', '#c0392b'),  # 红
            ('#4ecdc4', '#2ecc71'),  # 绿
            ('#45b7d1', '#2980b9'),  # 蓝
            ('#f39c12', '#d35400'),  # 橙
            ('#9b59b6', '#8e44ad'),  # 紫
            ('#1abc9c', '#16a085'),  # 青绿
        ]
        
        kpi_descriptions = {
            'max_pressure_mpa': '最大压力',
            'min_pressure_mpa': '最小压力',
            'pressure_spread_mpa': '压力波动幅度',
            'inlet_pressure_mpa': '入口压力',
            'outlet_pressure_mpa': '出口压力',
            'pressure_drop_mpa': '总压降',
            'max_temp_c': '最高温度',
            'min_temp_c': '最低温度',
            'inlet_temp_c': '入口温度',
            'max_velocity_ms': '最大流速',
            'min_velocity_ms': '最小流速',
            'max_density_kgm3': '最大密度',
            'min_density_kgm3': '最小密度',
            'total_time_s': '总仿真时长',
            'pipeline_length_km': '管线长度',
            'num_nodes': '计算节点数',
            'num_steps': '时间步数',
        }
        
        kpi_units = {
            'max_pressure_mpa': 'MPa', 'min_pressure_mpa': 'MPa',
            'inlet_pressure_mpa': 'MPa', 'outlet_pressure_mpa': 'MPa',
            'pressure_drop_mpa': 'MPa', 'pressure_spread_mpa': 'MPa',
            'max_temp_c': '°C', 'min_temp_c': '°C', 'inlet_temp_c': '°C',
            'max_velocity_ms': 'm/s', 'min_velocity_ms': 'm/s',
            'max_density_kgm3': 'kg/m³', 'min_density_kgm3': 'kg/m³',
            'total_time_s': 's', 'pipeline_length_km': 'km',
            'num_nodes': '个', 'num_steps': '步',
        }
        
        kpi_formats = {
            'max_pressure_mpa': '{:.2f}', 'min_pressure_mpa': '{:.2f}',
            'inlet_pressure_mpa': '{:.2f}', 'outlet_pressure_mpa': '{:.2f}',
            'pressure_drop_mpa': '{:.2f}', 'pressure_spread_mpa': '{:.2f}',
            'max_temp_c': '{:.1f}', 'min_temp_c': '{:.1f}', 'inlet_temp_c': '{:.1f}',
            'max_velocity_ms': '{:.2f}', 'min_velocity_ms': '{:.2f}',
            'max_density_kgm3': '{:.1f}', 'min_density_kgm3': '{:.1f}',
            'total_time_s': '{:.0f}', 'pipeline_length_km': '{:.1f}',
            'num_nodes': '{:.0f}', 'num_steps': '{:.0f}',
        }
        
        for i, (key, value) in enumerate(kpis.items()):
            if key == 'error':
                continue
            desc = kpi_descriptions.get(key, key)
            unit = kpi_units.get(key, '')
            fmt = kpi_formats.get(key, '{:.2f}')
            color = color_pairs[i % len(color_pairs)]
            
            if isinstance(value, (int, float)):
                formatted = fmt.format(value)
            else:
                formatted = str(value)
            
            cards.append(f'''
            <div style="background: linear-gradient(135deg, {color[0]}22, {color[1]}11);
                        border-left: 4px solid {color[0]};
                        border-radius: 8px; padding: 12px 16px; margin: 6px;
                        min-width: 140px; flex: 1;">
                <div style="font-size: 11px; color: #888; text-transform: uppercase;
                           letter-spacing: 0.5px; margin-bottom: 4px;">{desc}</div>
                <div style="font-size: 22px; font-weight: 700; color: {color[0]};
                           line-height: 1.2;">{formatted}
                    <span style="font-size: 12px; color: #999; font-weight: 400;"> {unit}</span>
                </div>
            </div>''')
        
        html = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif;
       background: #1a1a2e; color: #eee; padding: 20px; }}
h2 {{ margin-bottom: 16px; color: #4ecdc4; }}
.kpi-grid {{ display: flex; flex-wrap: wrap; gap: 6px; }}
</style></head><body>
<h2>📊 {title}</h2>
<div class="kpi-grid">{"".join(cards)}</div>
</body></html>'''
        
        return html
    
    @staticmethod
    def kpis_to_text(kpis: dict) -> str:
        """KPI → 纯文本报告"""
        lines = ['=' * 50, '📊 管道瞬态分析 KPI 报告', '=' * 50]
        sections = {
            '压力': ['max_pressure_mpa', 'min_pressure_mpa', 'pressure_spread_mpa',
                     'inlet_pressure_mpa', 'outlet_pressure_mpa', 'pressure_drop_mpa'],
            '温度': ['max_temp_c', 'min_temp_c', 'inlet_temp_c'],
            '流动': ['max_velocity_ms', 'min_velocity_ms', 'max_density_kgm3'],
            '系统': ['total_time_s', 'pipeline_length_km', 'num_nodes', 'num_steps'],
        }
        
        for section, keys in sections.items():
            vals = [(k, kpis.get(k)) for k in keys if k in kpis]
            if vals:
                lines.append(f'\n▸ {section}')
                for k, v in vals:
                    if isinstance(v, float):
                        lines.append(f'  {k:25s} {v:>10.3f}')
                    else:
                        lines.append(f'  {k:25s} {str(v):>10s}')
        
        return '\n'.join(lines)


# ─── 沿程分布 — 任意变量沿管线距离的剖面 ────────────────

class PipelineProfile:
    """管道沿程剖面 — 沿线里程数据可视化"""
    
    @staticmethod
    def along_pipe(result, variable='P', time_idx=-1, 
                    output_path=None, title='沿管道里程分布', 
                    backend='plotly'):
        """
        绘制沿管道里程的变量分布
        
        Args:
            result: TransientResult对象
            variable: 'P'(压力) / 'T'(温度) / 'v'(流速) / 'rho'(密度)
            time_idx: 时刻索引（-1=最终时刻）
            output_path: HTML/PNG输出路径
            title: 图表标题
            backend: 'plotly' / 'matplotlib'
        """
        var_map = {'P': ('压力', 'MPa', 1e6),
                   'T': ('温度', '°C', 1.0),
                   'v': ('流速', 'm/s', 1.0),
                   'rho': ('密度', 'kg/m³', 1.0)}
        
        if variable not in var_map:
            return f"[不支持的变量: {variable}]"
        
        varname, unit, scale = var_map[variable]
        data_arr = getattr(result, variable, None)
        x_arr = getattr(result, 'x', None)
        t_arr = getattr(result, 't', None)
        
        if data_arr is None or x_arr is None:
            return f"[缺少数据: {variable} 或 x]"
        
        values = data_arr[time_idx] / scale
        x_km = x_arr / 1000
        
        # 温度补偿
        if variable == 'T':
            values = values - 273.15
        
        if backend == 'plotly':
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_km, y=values,
                mode='lines+markers',
                name=f't={t_arr[time_idx]:.1f}s' if t_arr is not None else '',
                line=dict(width=3, color='#4ecdc4'),
                marker=dict(size=4, color='#45b7d1'),
            ))
            
            # 填充面积
            fig.add_trace(go.Scatter(
                x=list(x_km) + list(x_km)[::-1],
                y=list(values) + [0]*len(values),
                fill='tozerox', fillcolor='rgba(78,205,196,0.15)',
                line=dict(width=0), showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.update_layout(
                title=dict(text=f'{title}: {varname}', font=dict(size=18)),
                xaxis=dict(title='管道里程 (km)', gridcolor='#333', 
                          tickfont=dict(size=12)),
                yaxis=dict(title=f'{varname} ({unit})', gridcolor='#333',
                          tickfont=dict(size=12)),
                template='plotly_dark',
                hovermode='x unified',
                margin=dict(l=60, r=40, t=60, b=60),
                height=450,
            )
            
            if output_path:
                fig.write_html(output_path)
            return fig
        
        else:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(x_km, values, 'b-', linewidth=2, label=f't={t_arr[time_idx]:.1f}s' if t_arr is not None else '')
            ax.fill_between(x_km, 0, values, alpha=0.1)
            ax.set_xlabel('距离 (km)', fontsize=12)
            ax.set_ylabel(f'{varname} ({unit})', fontsize=12)
            ax.set_title(f'{title}: {varname}')
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()
            if output_path:
                plt.savefig(output_path, dpi=150)
            return fig
    
    @staticmethod
    def multi_variable_profile(result, variables=None, time_idx=-1,
                                 output_path=None):
        """多变量沿程对比剖面（同一个图多个变量归一化后对比）"""
        if variables is None:
            variables = ['P', 'T', 'v']
        
        norm_data = {}
        x_arr = getattr(result, 'x', None)
        if x_arr is None:
            return "[无x数据]"
        x_km = x_arr / 1000
        
        for var in variables:
            arr = getattr(result, var, None)
            if arr is None:
                continue
            vals = arr[time_idx]
            # 归一化
            vmin, vmax = np.min(vals), np.max(vals)
            if vmax - vmin > 0:
                vals_norm = (vals - vmin) / (vmax - vmin)
            else:
                vals_norm = vals * 0 + 0.5
            norm_data[var] = vals_norm
        
        if not norm_data:
            return "[无可处理变量]"
        
        import plotly.graph_objects as go
        colors = {'P': '#ff6b6b', 'T': '#f39c12', 'v': '#4ecdc4', 'rho': '#9b59b6'}
        labels = {'P': '压力', 'T': '温度', 'v': '流速', 'rho': '密度'}
        
        fig = go.Figure()
        for var, vals in norm_data.items():
            fig.add_trace(go.Scatter(
                x=x_km, y=vals, mode='lines',
                name=labels.get(var, var),
                line=dict(width=2.5, color=colors.get(var, '#fff')),
            ))
        
        fig.update_layout(
            title='多变量沿程归一化对比',
            xaxis=dict(title='管道里程 (km)'),
            yaxis=dict(title='归一化值 (0-1)'),
            template='plotly_dark',
            hovermode='x unified',
        )
        
        if output_path:
            fig.write_html(output_path)
        return fig
    
    @staticmethod
    def time_evolution_at_point(result, x_location_m: float,
                                  variables: List[str] = None,
                                  output_path: str = None,
                                  title: str = '指定位置时程曲线'):
        """
        指定管道位置随时间的变化曲线
        
        Args:
            result: TransientResult
            x_location_m: 位置（米）
            variables: 变量列表 ['P', 'T', 'v']
            output_path: 输出路径
            title: 标题
        """
        if variables is None:
            variables = ['P', 'T', 'v']
        
        t_arr = getattr(result, 't', None)
        x_arr = getattr(result, 'x', None)
        if t_arr is None or x_arr is None:
            return "[缺少时间/位置数据]"
        
        # 找到最近的节点
        idx = np.argmin(np.abs(x_arr - x_location_m))
        actual_x = x_arr[idx]
        
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        fig = make_subplots(rows=len(variables), cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.08,
                            subplot_titles=[f'位置: {actual_x/1000:.2f} km'])
        
        colors = {'P': '#ff6b6b', 'T': '#f39c12', 'v': '#4ecdc4', 'rho': '#9b59b6'}
        labels = {'P': ('压力 (MPa)', 1e6),
                  'T': ('温度 (°C)', 1.0),
                  'v': ('流速 (m/s)', 1.0),
                  'rho': ('密度 (kg/m³)', 1.0)}
        
        for i, var in enumerate(variables):
            arr = getattr(result, var, None)
            if arr is None:
                continue
            
            unit_label, scale = labels.get(var, (var, 1))
            values = arr[:, idx] / scale
            if var == 'T':
                values = values - 273.15
            
            fig.add_trace(
                go.Scatter(x=t_arr, y=values, mode='lines',
                          name=labels.get(var, var)[0].split(' (')[0],
                          line=dict(color=colors.get(var, '#fff'), width=2.5)),
                row=i+1, col=1
            )
            
            fig.update_yaxes(title_text=unit_label, row=i+1, col=1,
                            gridcolor='#333')
        
        fig.update_xaxes(title_text='时间 (s)', row=len(variables), col=1,
                        gridcolor='#333')
        fig.update_layout(
            title=title,
            template='plotly_dark',
            height=250 * len(variables) + 100,
            hovermode='x unified',
            showlegend=False,
        )
        
        if output_path:
            fig.write_html(output_path)
        return fig
    
    @staticmethod
    def multi_point_comparison(result, x_locations_m: List[float],
                                 variable='P', output_path=None):
        """
        多点时间曲线对比（同一变量, 不同位置）
        
        Args:
            result: TransientResult
            x_locations_m: 位置列表 [1000, 5000, 10000]
            variable: 变量 'P' / 'T' / 'v'
            output_path: 输出路径
        """
        t_arr = getattr(result, 't', None)
        x_arr = getattr(result, 'x', None)
        arr = getattr(result, variable, None)
        
        if t_arr is None or x_arr is None or arr is None:
            return "[缺少数据]"
        
        labels = {'P': '压力 (MPa)', 'T': '温度 (°C)', 'v': '流速 (m/s)'}
        scales = {'P': 1e6, 'T': 1.0, 'v': 1.0}
        scale = scales.get(variable, 1.0)
        
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f39c12', '#9b59b6']
        
        import plotly.graph_objects as go
        fig = go.Figure()
        
        for i, x_loc in enumerate(x_locations_m):
            idx = np.argmin(np.abs(x_arr - x_loc))
            actual_x = x_arr[idx]
            values = arr[:, idx] / scale
            if variable == 'T':
                values = values - 273.15
            
            fig.add_trace(go.Scatter(
                x=t_arr, y=values, mode='lines',
                name=f'{actual_x/1000:.1f} km',
                line=dict(color=colors[i % len(colors)], width=2.5),
            ))
        
        fig.update_layout(
            title=f'{labels.get(variable, variable)} — 多位置对比',
            xaxis=dict(title='时间 (s)', gridcolor='#333'),
            yaxis=dict(title=labels.get(variable, variable), gridcolor='#333'),
            template='plotly_dark',
            hovermode='x unified',
        )
        
        if output_path:
            fig.write_html(output_path)
        return fig


# ─── 时空全貌 (Heatmap) ──────────────────────────────────

class PipelineHeatmap:
    """管道时空热力图 — 距离×时间全貌"""
    
    @staticmethod
    def spacetime_heatmap(result, variable='P', output_path=None,
                           title=None):
        """
        时空热力图 (距离×时间)
        
        Args:
            result: TransientResult
            variable: 'P' / 'T' / 'v' / 'rho'
            output_path: 输出路径
            title: 标题
        """
        arr = getattr(result, variable, None)
        x_arr = getattr(result, 'x', None)
        t_arr = getattr(result, 't', None)
        
        if arr is None:
            return f"[无{variable}数据]"
        
        labels = {'P': 'Pressure (MPa)', 'T': 'Temperature (°C)',
                  'v': 'Velocity (m/s)', 'rho': 'Density (kg/m³)'}
        scales = {'P': 1e6, 'T': 1.0, 'v': 1.0, 'rho': 1.0}
        scale = scales.get(variable, 1.0)
        
        data = arr.T / scale  # (nx, nt)
        if variable == 'T':
            data = data - 273.15
        
        import plotly.graph_objects as go
        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=t_arr,
            y=x_arr / 1000,
            colorscale='RdBu_r',
            colorbar=dict(title=labels.get(variable, variable)),
            hovertemplate='时间: %{x:.1f}s<br>距离: %{y:.1f}km<br>值: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title or f'{labels.get(variable, variable)} — 时空分布',
            xaxis=dict(title='时间 (s)', gridcolor='#444'),
            yaxis=dict(title='管道里程 (km)', gridcolor='#444'),
            template='plotly_dark',
            height=500,
        )
        
        if output_path:
            fig.write_html(output_path)
        return fig
    
    @staticmethod
    def animation_profile(result, variable='P', output_path=None):
        """沿程分布随时间变化的动画（仅plotly）"""
        arr = getattr(result, variable, None)
        x_arr = getattr(result, 'x', None)
        t_arr = getattr(result, 't', None)
        
        if arr is None or x_arr is None:
            return "[缺少数据]"
        
        scales = {'P': 1e6, 'T': 1.0, 'v': 1.0, 'rho': 1.0}
        labels = {'P': 'Pressure (MPa)', 'T': 'Temp (°C)',
                  'v': 'Velocity (m/s)', 'rho': 'Density (kg/m³)'}
        scale = scales.get(variable, 1.0)
        
        import plotly.graph_objects as go
        x_km = x_arr / 1000
        
        frames = []
        for i in range(min(100, len(t_arr))):
            idx = int(i * len(t_arr) / 100) if len(t_arr) > 100 else i
            vals = arr[idx] / scale
            if variable == 'T':
                vals = vals - 273.15
            
            frames.append(go.Frame(
                data=[go.Scatter(x=x_km, y=vals, mode='lines+markers',
                                line=dict(width=3, color='#4ecdc4'))],
                name=f'{t_arr[idx]:.1f}s'
            ))
        
        fig = go.Figure(
            data=[go.Scatter(x=x_km, y=arr[0]/scale, mode='lines+markers',
                            line=dict(width=3, color='#4ecdc4'))],
            layout=go.Layout(
                title=f'{labels.get(variable, variable)} 动态演变',
                xaxis=dict(title='管道里程 (km)', range=[0, max(x_km)]),
                yaxis=dict(title=labels.get(variable, variable),
                          range=[0, max(arr[-1]/scale) * 1.1]),
                template='plotly_dark',
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [
                        {'label': '▶ Play', 'method': 'animate',
                         'args': [None, {'frame': {'duration': 100, 'redraw': True},
                                        'fromcurrent': True}]},
                        {'label': '⏸ Pause', 'method': 'animate',
                         'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                                          'mode': 'immediate'}]},
                    ]
                }],
                sliders=[{
                    'steps': [{'args': [[f.name], {'frame': {'duration': 0, 'redraw': True},
                                                   'mode': 'immediate'}],
                               'label': f.name, 'method': 'animate'}
                              for f in frames],
                    'currentvalue': {'prefix': '时间: ', 'font': {'size': 14}},
                }]
            ),
            frames=frames
        )
        
        if output_path:
            fig.write_html(output_path)
        return fig


# ─── CSV/数据导出 ──────────────────────────────────────

class PipelineDataExport:
    """管道数据导出（CSV/JSON/报告）"""
    
    @staticmethod
    def to_csv(result, variables=None, output_path='pipeline_data.csv'):
        """导出管道数据为CSV"""
        if variables is None:
            variables = ['P', 'T', 'v', 'rho']
        
        x_arr = getattr(result, 'x', None)
        t_arr = getattr(result, 't', None)
        
        if x_arr is None or t_arr is None:
            return "[缺少位置/时间数据]"
        
        # 最终时刻沿程数据
        header = ['distance_m', 'distance_km']
        for var in variables:
            arr = getattr(result, var, None)
            if arr is not None:
                header.append(var)
        
        rows = [','.join(header)]
        for i in range(len(x_arr)):
            row = [f'{x_arr[i]:.2f}', f'{x_arr[i]/1000:.4f}']
            for var in variables:
                arr = getattr(result, var, None)
                if arr is not None:
                    val = arr[-1, i]
                    if var == 'T':
                        val = val - 273.15
                    row.append(f'{val:.6f}')
            rows.append(','.join(row))
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(rows))
        
        return output_path


# ─── 导出 ──────────────────────────────────────────────

__all__ = [
    'PipelineDataPanel',
    'PipelineProfile',
    'PipelineHeatmap',
    'PipelineDataExport',
]


if __name__ == '__main__':
    print("=" * 55)
    print("  PipelineSim 数据处理与可视化 v1.0")
    print("  ├─ PipelineDataPanel   — KPI指标卡面板")
    print("  ├─ PipelineProfile     — 沿程分布曲线")
    print("  │   ├─ along_pipe       — 单变量沿程剖面")
    print("  │   ├─ multi_variable_profile — 多变量对比")
    print("  │   ├─ time_evolution_at_point — 单点时程")
    print("  │   └─ multi_point_comparison  — 多点对比")
    print("  ├─ PipelineHeatmap     — 时空热力图/动画")
    print("  ├─ PipelineDataExport  — CSV导出")
    print("  └─ 集成 PipelineSim 仪表板")
    print("=" * 55)
    
    # 快速生成测试
    print("\n生成测试数据...")
    import numpy as np
    from sim.pipe import Pipe
    from sim.fluid import Liquid
    from sim.solver import SinglePhaseTransientSolver, TransientResult
    
    pipe = Pipe(length=10000, diameter=0.5, roughness=0.05)
    fluid = Liquid(rho0=1000, K=2.0e9, mu=0.001)
    solver = SinglePhaseTransientSolver(pipe, fluid)
    
    from sim.solver import flow_inlet, pressure_outlet
    bc_in = flow_inlet(Q=lambda t: 1.0 if t < 10 else 0.5)
    bc_out = pressure_outlet(P=lambda t: 2.0e6)
    
    result = solver.solve(100, 20, bc_in, bc_out)
    
    print(f"\n结果形状: P={getattr(result, 'P', 'N/A').shape}")
    
    # KPI
    print("\n📊 KPI:")
    kpis = PipelineDataPanel.compute_kpis(result)
    for k, v in list(kpis.items())[:8]:
        print(f"  {k}: {v:.3f}")
    
    print("\n✅ Pipeline Data Processing Skill 就绪!")
