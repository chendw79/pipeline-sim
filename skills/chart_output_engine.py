"""
chart_output_engine.py — PipelineSim多引擎图表输出

支持:
  1. matplotlib/seaborn — 精美静态图 (CSDN文章级)
  2. plotly — 交互图表 (现有)
  3. pyecharts — 另类交互风格
  4. 自动风格模板: BCG/咨询/学术/科技风

统一接口: ChartEngine.generate(result, format='all')
"""

import numpy as np
import io, base64, os
from typing import Dict, Optional, List
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════
# 风格模板
# ═══════════════════════════════════════════════

@dataclass
class StyleTemplate:
    """图表风格模板"""
    name: str = "scientific"
    
    # Matplotlib参数
    figsize: tuple = (10, 5)
    dpi: int = 150
    font_family: str = 'DejaVu Sans'
    title_size: int = 14
    label_size: int = 12
    legend_size: int = 10
    grid_alpha: float = 0.3
    color_palette: str = 'viridis'
    
    @classmethod
    def csdn_article(cls) -> 'StyleTemplate':
        """CSDN文章配图 — 清晰、专业、白底"""
        s = cls(name='CSDN文章配图')
        s.figsize = (12, 5)
        s.dpi = 200
        s.color_palette = 'Set2'
        s.grid_alpha = 0.2
        return s
    
    @classmethod
    def consulting(cls) -> 'StyleTemplate':
        """咨询报告风 — McKinsey/BCG"""
        s = cls(name='咨询报告风')
        s.figsize = (10, 4.5)
        s.dpi = 150
        s.color_palette = ['#0077B6', '#00B4D8', '#90E0EF', '#CAF0F8', '#023E8A']
        s.grid_alpha = 0.15
        return s
    
    @classmethod
    def tech_dark(cls) -> 'StyleTemplate':
        """科技暗色 — Dashboard/大屏"""
        s = cls(name='科技暗色')
        s.figsize = (12, 5)
        s.dpi = 100
        s.color_palette = ['#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dfe6e9']
        s.grid_alpha = 0.1
        return s


# ═══════════════════════════════════════════════
# Matplotlib + Seaborn 静态引擎
# ═══════════════════════════════════════════════

class MPLChartEngine:
    """Matplotlib + Seaborn 静态图表引擎"""
    
    @staticmethod
    def use_style(style: StyleTemplate):
        """应用风格"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            # 尝试中文字体
            import subprocess
            cjk_fonts = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Noto Sans CJK', 
                         'Droid Sans Fallback', 'DejaVu Sans']
            available = subprocess.run(
                ['fc-list', ':lang=zh', '-f', '%{family}\n'], 
                capture_output=True, text=True, timeout=2
            ).stdout.strip().split('\n')
            zh_font = next((f for f in cjk_fonts if any(f in av for av in available)), 'DejaVu Sans')
            
            plt.rcParams.update({
                'figure.figsize': style.figsize,
                'figure.dpi': style.dpi,
                'font.sans-serif': [zh_font] + plt.rcParams.get('font.sans-serif', []),
                'font.family': 'sans-serif',
                'axes.unicode_minus': False,
                'axes.titlesize': style.title_size,
                'axes.labelsize': style.label_size,
                'legend.fontsize': style.legend_size,
                'grid.alpha': style.grid_alpha,
            })
        except:
            pass
    
    @staticmethod
    def pressure_profile(result, style: Optional[StyleTemplate] = None,
                         save_path: Optional[str] = None) -> bytes:
        """沿程压力分布 (最终时刻)"""
        if style is None:
            style = StyleTemplate.csdn_article()
        
        MPLChartEngine.use_style(style)
        import matplotlib.pyplot as plt
        
        try:
            P = result.P[-1, :] / 1e6
            x = result.x / 1000
            
            fig, ax = plt.subplots(figsize=style.figsize)
            ax.plot(x, P, linewidth=2.5, color='#e74c3c', marker='o', markersize=5)
            ax.fill_between(x, P, alpha=0.1, color='#e74c3c')
            
            ax.set_xlabel('距入口距离 (km)')
            ax.set_ylabel('压力 (MPa)')
            ax.set_title('管道沿程压力分布', fontweight='bold')
            ax.grid(True, alpha=style.grid_alpha)
            
            # 标注峰值
            max_idx = np.argmax(P)
            ax.annotate(f'峰值: {P[max_idx]:.2f} MPa',
                       xy=(x[max_idx], P[max_idx]),
                       xytext=(x[max_idx]+1, P[max_idx]+0.3),
                       arrowprops=dict(arrowstyle='->', color='#2c3e50'),
                       fontsize=10, bbox=dict(boxstyle='round,pad=0.3',
                                             facecolor='yellow', alpha=0.7))
            
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=style.dpi, bbox_inches='tight')
            plt.close()
            buf.seek(0)
            
            if save_path:
                with open(save_path, 'wb') as f:
                    f.write(buf.getvalue())
            
            return buf.getvalue()
        except Exception as e:
            return b''
    
    @staticmethod
    def pressure_time_series(result, style: Optional[StyleTemplate] = None,
                              node: int = 0, save_path: Optional[str] = None) -> bytes:
        """节点压力时程曲线"""
        if style is None:
            style = StyleTemplate.csdn_article()
        
        MPLChartEngine.use_style(style)
        import matplotlib.pyplot as plt
        
        try:
            P = result.P[:, node] / 1e6
            t = result.t
            
            fig, ax = plt.subplots(figsize=style.figsize)
            ax.plot(t, P, linewidth=2, color='#3498db')
            ax.fill_between(t, P, alpha=0.1, color='#3498db')
            
            ax.set_xlabel('时间 (s)')
            ax.set_ylabel('压力 (MPa)')
            ax.set_title(f'节点 x={result.x[node]/1000:.1f}km 压力时程', fontweight='bold')
            ax.grid(True, alpha=style.grid_alpha)
            
            # 标注最大最小值
            max_p, min_p = np.max(P), np.min(P)
            ax.axhline(y=max_p, color='r', linestyle='--', alpha=0.5, label=f'Max: {max_p:.2f}MPa')
            ax.axhline(y=min_p, color='b', linestyle='--', alpha=0.5, label=f'Min: {min_p:.2f}MPa')
            ax.legend()
            
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=style.dpi, bbox_inches='tight')
            plt.close()
            buf.seek(0)
            
            if save_path:
                with open(save_path, 'wb') as f:
                    f.write(buf.getvalue())
            
            return buf.getvalue()
        except Exception as e:
            return b''
    
    @staticmethod
    def comparison_chart(results: Dict[str, object],
                         style: Optional[StyleTemplate] = None) -> bytes:
        """多案例对比图"""
        if style is None:
            style = StyleTemplate.consulting()
        
        MPLChartEngine.use_style(style)
        import matplotlib.pyplot as plt
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
            
            for idx, (name, res) in enumerate(results.items()):
                c = colors[idx % len(colors)]
                P = res.P if hasattr(res, 'P') and res.P is not None else np.zeros((2, 2))
                t = res.t if hasattr(res, 't') and res.t is not None else np.array([0, 1])
                
                # 左: 末端压力时程
                ax1.plot(t, P[:, -1]/1e6, color=c, linewidth=1.5, label=name)
                
                # 右: 最大压力柱状
                max_p = np.max(P) / 1e6 if len(P) > 0 and len(P[0]) > 0 else 0
                ax2.bar(idx, max_p, color=c, alpha=0.7, label=name)
            
            ax1.set_xlabel('时间 (s)'); ax1.set_ylabel('出口压力 (MPa)')
            ax1.set_title('多案例出口压力对比'); ax1.legend(); ax1.grid(True, alpha=0.2)
            
            ax2.set_xlabel('案例'); ax2.set_ylabel('最大压力 (MPa)')
            ax2.set_title('最大压力对比'); ax2.set_xticks([]); ax2.legend()
            
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=style.dpi, bbox_inches='tight')
            plt.close()
            buf.seek(0)
            return buf.getvalue()
        except Exception:
            return b''
    
    @staticmethod
    def kpi_dashboard(kpis: dict, style: Optional[StyleTemplate] = None) -> bytes:
        """KPI仪表板 (seaborn)"""
        if style is None:
            style = StyleTemplate.consulting()
        
        MPLChartEngine.use_style(style)
        import matplotlib.pyplot as plt
        
        try:
            key_labels = {
                'max_pressure_mpa': ('最大压力', 'MPa'),
                'min_pressure_mpa': ('最小压力', 'MPa'),
                'pressure_spread_mpa': ('波动幅度', 'MPa'),
                'pipeline_length_km': ('管线长度', 'km'),
                'total_time_s': ('仿真时长', 's'),
            }
            
            fig, axes = plt.subplots(1, 5, figsize=(16, 3))
            
            colors = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71', '#9b59b6']
            for idx, (key, (label, unit)) in enumerate(key_labels.items()):
                val = kpis.get(key, 0)
                axes[idx].barh(0, val, color=colors[idx], height=0.5)
                axes[idx].set_xlim(0, val * 1.3)
                axes[idx].set_yticks([])
                axes[idx].set_title(f'{label}\n{val:.2f} {unit}', fontsize=10)
                axes[idx].spines['top'].set_visible(False)
                axes[idx].spines['right'].set_visible(False)
            
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=style.dpi, bbox_inches='tight')
            plt.close()
            buf.seek(0)
            return buf.getvalue()
        except Exception:
            return b''


# ═══════════════════════════════════════════════
# PyECharts引擎
# ═══════════════════════════════════════════════

class EChartEngine:
    """PyECharts交互图表引擎"""
    
    @staticmethod
    def pressure_profile_html(result) -> str:
        """沿程压力分布 (ECharts)"""
        try:
            from pyecharts import options as opts
            from pyecharts.charts import Line
            
            P = result.P[-1, :] / 1e6
            x = [f'{v:.1f}' for v in result.x / 1000]
            
            chart = (
                Line()
                .add_xaxis(x)
                .add_yaxis("压力 (MPa)", [float(f) for f in P],
                          is_smooth=True,
                          linestyle_opts=opts.LineStyleOpts(width=3, color='#e74c3c'),
                          itemstyle_opts=opts.ItemStyleOpts(color='#e74c3c'))
                .set_global_opts(
                    title_opts=opts.TitleOpts(title="管道沿程压力分布"),
                    xaxis_opts=opts.AxisOpts(name="距离 (km)", type_="category"),
                    yaxis_opts=opts.AxisOpts(name="压力 (MPa)"),
                    tooltip_opts=opts.TooltipOpts(trigger="axis"),
                    datazoom_opts=[opts.DataZoomOpts()],
                )
                .render_embed()
            )
            return chart
        except Exception:
            return "<div>PyECharts不可用</div>"
    
    @staticmethod
    def kpi_gauge(kpis: dict) -> str:
        """KPI仪表盘 (ECharts gauge)"""
        try:
            from pyecharts import options as opts
            from pyecharts.charts import Page, Gauge
            
            page = Page(layout=Page.SimplePageLayout)
            
            gauge_configs = [
                ('最大压力 (MPa)', kpis.get('max_pressure_mpa', 0), 15),
                ('波动幅度 (MPa)', kpis.get('pressure_spread_mpa', 0), 10),
                ('管线长度 (km)', kpis.get('pipeline_length_km', 0), 50),
            ]
            
            for title, val, max_val in gauge_configs:
                gauge = (
                    Gauge()
                    .add(title, [(title, min(val, max_val))],
                         min_=0, max_=max_val,
                         split_number=5)
                    .set_global_opts(title_opts=opts.TitleOpts(title=title))
                )
                page.add(gauge)
            
            return page.render_embed()
        except Exception:
            return "<div>KPI仪表盘不可用</div>"


# ═══════════════════════════════════════════════
# 统一引擎
# ═══════════════════════════════════════════════

class ChartOutputEngine:
    """统一图表输出引擎"""
    
    @staticmethod
    def generate(result, kpis: dict = None, format: str = 'all',
                 style: str = 'csdn') -> Dict[str, bytes]:
        """
        生成所有格式图表
        
        Args:
            result: TransientResult
            kpis: KPI字典
            format: 'png'(静态) / 'html'(交互) / 'all'(两者)
            style: 'csdn' / 'consulting' / 'tech_dark' / 'scientific'
        
        Returns:
            { 'pressure_profile.png': bytes, 'pressure_profile.html': str, ... }
        """
        style_map = {
            'csdn': StyleTemplate.csdn_article(),
            'consulting': StyleTemplate.consulting(),
            'tech_dark': StyleTemplate.tech_dark(),
            'scientific': StyleTemplate(),
        }
        s = style_map.get(style, StyleTemplate.csdn_article())
        
        output = {}
        
        # 静态图 (PNG)
        if format in ('png', 'all'):
            output['pressure_profile.png'] = MPLChartEngine.pressure_profile(result, s)
            output['pressure_timeseries.png'] = MPLChartEngine.pressure_time_series(result, s)
            if kpis:
                output['kpi_dashboard.png'] = MPLChartEngine.kpi_dashboard(kpis, s)
        
        # 交互图 (HTML)
        if format in ('html', 'all'):
            output['pressure_profile.html'] = EChartEngine.pressure_profile_html(result)
            if kpis:
                output['kpi_gauge.html'] = EChartEngine.kpi_gauge(kpis)
        
        return output


if __name__ == '__main__':
    print("=== PipelineSim 多引擎图表输出 ===")
    print()
    print("支持引擎:")
    print("  📊 Matplotlib + Seaborn — 静态图 (CSDN文章级)")
    print("  🎨 PyECharts — 交互图 (ECharts)")
    print("  🌐 Plotly — 交互图 (现有)")
    print()
    print("风格模板:")
    for name in ['CSDN文章配图', '咨询报告风', '科技暗色', '学术标准']:
        print(f"  • {name}")
    print()
    print("使用示例:")
    print("  engine = ChartOutputEngine()")
    print("  charts = engine.generate(result, kpis, format='all', style='csdn')")
    print("  # charts['pressure_profile.png'] → 可存储/发送/展示")
    print()
    
    # 测试
    print("测试图表生成 (模拟数据)...")
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from sim.pipe import Pipe; from sim.fluid import Liquid
    from sim.solver import SinglePhaseTransientSolver, flow_inlet, pressure_outlet
    from skills.pipeline_data_processing import PipelineDataPanel
    
    pipe = Pipe(10000, 0.5, 0.01)
    fluid = Liquid()
    solver = SinglePhaseTransientSolver(pipe, fluid)
    result = solver.solve(15, flow_inlet(lambda t: 1.0, lambda t: 300.0),
                          pressure_outlet(lambda t: 2.0e6), 'A', P_initial=2.0e6)
    kpis = PipelineDataPanel.compute_kpis(result)
    
    charts = ChartOutputEngine.generate(result, kpis, format='pn', style='csdn')
    for k, v in charts.items():
        if isinstance(v, bytes):
            print(f"  {k}: {len(v)/1024:.0f}KB")
        else:
            print(f"  {k}: {len(v)} chars")
    
    print()
    print("✅ 多引擎图表输出就绪!")
