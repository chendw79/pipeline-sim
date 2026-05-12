"""
report_generator.py — PipelineSim HTML→PDF专业报告引擎

技术栈: Jinja2 (模板) + WeasyPrint (HTML→PDF)
对比reportlab: 更灵活、支持CSS布局、可内嵌图表
"""

import os, io, base64, json
from datetime import datetime
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════
# CSS样式（内置）
# ═══════════════════════════════════════════════

REPORT_CSS = """
@page {
    size: A4;
    margin: 2cm 2.5cm;
    @top-center {
        content: element(header);
        font-size: 8pt;
        color: #666;
    }
    @bottom-center {
        content: "第 " counter(page) " 页";
        font-size: 8pt;
        color: #666;
    }
}
body {
    font-family: "DejaVu Sans", sans-serif;
    font-size: 10pt;
    line-height: 1.6;
    color: #333;
}
h1 { font-size: 20pt; color: #1a1a2e; border-bottom: 3px solid #4ecdc4; padding-bottom: 8px; }
h2 { font-size: 14pt; color: #2c3e50; margin-top: 20px; padding-left: 8px;
     border-left: 4px solid #4ecdc4; }
h3 { font-size: 11pt; color: #34495e; margin-top: 15px; }
table { width: 100%; border-collapse: collapse; margin: 10px 0; }
th { background: #2c3e50; color: white; padding: 8px 10px; text-align: left; font-size: 9pt; }
td { padding: 6px 10px; border-bottom: 1px solid #ddd; font-size: 9pt; }
tr:nth-child(even) { background: #f8f9fa; }
.kpi-card { display: inline-block; width: 30%; margin: 5px; padding: 10px;
            border-radius: 6px; border-left: 4px solid #4ecdc4;
            background: #f0f7f6; }
.kpi-label { font-size: 8pt; color: #666; }
.kpi-value { font-size: 16pt; font-weight: bold; color: #1a1a2e; }
.kpi-unit { font-size: 8pt; color: #999; }
.warning { background: #fff3cd; border: 1px solid #ffc107; padding: 8px 12px;
           border-radius: 4px; margin: 8px 0; }
.danger { background: #f8d7da; border: 1px solid #dc3545; padding: 8px 12px;
          border-radius: 4px; margin: 8px 0; }
.success { background: #d4edda; border: 1px solid #28a745; padding: 8px 12px;
           border-radius: 4px; margin: 8px 0; }
.info { background: #d1ecf1; border: 1px solid #17a2b8; padding: 8px 12px;
        border-radius: 4px; margin: 8px 0; }
img { max-width: 100%; margin: 10px 0; border: 1px solid #e0e0e0; border-radius: 4px; }
.footer { margin-top: 30px; padding-top: 10px; border-top: 1px solid #ddd;
          font-size: 8pt; color: #999; text-align: center; }
.header-title { font-size: 10pt; color: #666; text-align: center; }
.page-break { page-break-before: always; }
"""


# ═══════════════════════════════════════════════
# 报告模板
# ═══════════════════════════════════════════════

REPORT_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>{{ css }}</style>
</head>
<body>

<h1>{{ title }}</h1>
<p style="color:#666;font-size:9pt;">
生成时间: {{ generated_at }} | PipelineSim v2.0
{% if project_name %} | {{ project_name }}{% endif %}
</p>

<!-- KPI摘要 -->
<h2>KPI 关键指标</h2>
<div style="text-align:center;">
{% for kpi in kpi_cards %}
<div class="kpi-card">
    <div class="kpi-label">{{ kpi.label }}</div>
    <div class="kpi-value">{{ kpi.value }} <span class="kpi-unit">{{ kpi.unit }}</span></div>
</div>
{% endfor %}
</div>

<!-- KPI详细表格 -->
<h3>详细指标</h3>
<table>
<tr><th>指标</th><th>数值</th><th>单位</th><th>说明</th></tr>
{% for kpi in kpi_table %}
<tr><td>{{ kpi.name }}</td><td>{{ kpi.value }}</td><td>{{ kpi.unit }}</td><td>{{ kpi.desc }}</td></tr>
{% endfor %}
</table>

<!-- 自动发现 -->
{% if findings %}
<h2>自动分析发现</h2>
{% for f in findings %}
<div class="{{ f.severity }}">
    <strong>{{ f.variable }}: {{ f.value }}</strong><br>
    {{ f.message }}<br>
    {% if f.advice %}<em>建议: {{ f.advice }}</em>{% endif %}
</div>
{% endfor %}
{% endif %}

<!-- 图表 -->
{% if charts_png %}
<h2>仿真结果图表</h2>
{% for name, img_b64 in charts_png.items() %}
<h3>{{ name }}</h3>
<img src="data:image/png;base64,{{ img_b64 }}" alt="{{ name }}">
{% endfor %}
{% endif %}

<!-- 仿真参数 -->
{% if config %}
<div class="page-break"></div>
<h2>仿真参数</h2>
<table>
<tr><th>参数</th><th>值</th></tr>
{% for key, val in config.items() %}
<tr><td>{{ key }}</td><td>{{ val }}</td></tr>
{% endfor %}
</table>
{% endif %}

<div class="footer">
PipelineSim — 管道瞬态仿真智能引擎 | 由AI Agent自动生成
</div>

</body>
</html>"""


# ═══════════════════════════════════════════════
# 报告生成器
# ═══════════════════════════════════════════════

class ReportGenerator:
    """HTML → PDF 专业报告"""
    
    @staticmethod
    def generate(result, kpis: dict = None, findings: dict = None,
                 charts: Dict[str, bytes] = None,
                 config: dict = None,
                 title: str = "管道瞬态分析报告",
                 project_name: str = "",
                 output_path: Optional[str] = None) -> str:
        """
        生成HTML/PDF报告
        
        Args:
            result: TransientResult
            kpis: KPI字典
            findings: auto_discover结果
            charts: {name: png_bytes} 图表
            config: 仿真配置参数
            title: 报告标题
            project_name: 项目名称
            output_path: 输出PDF路径
        
        Returns:
            output_path
        """
        if output_path is None:
            output_path = f'/tmp/pipeline_report_{int(datetime.now().timestamp())}.pdf'
        
        # 准备KPI
        kpi_cards = []
        kpi_table = []
        
        kpi_defs = [
            ('max_pressure_mpa', '最大压力', 'MPa', '管道最高压力值', True),
            ('min_pressure_mpa', '最小压力', 'MPa', '管道最低压力值', True),
            ('pressure_spread_mpa', '波动幅度', 'MPa', '压力波动范围', True),
            ('inlet_pressure_mpa', '入口压力', 'MPa', '管道入口端压力', False),
            ('outlet_pressure_mpa', '出口压力', 'MPa', '管道出口端压力', False),
            ('pipeline_length_km', '管线长度', 'km', '管道总长度', True),
            ('total_time_s', '仿真时长', 's', '仿真计算总时间', False),
            ('max_temp_c', '最高温度', '°C', '温度峰值', False),
        ]
        
        if kpis:
            for key, name, unit, desc, is_card in kpi_defs:
                if key in kpis:
                    val = kpis[key]
                    val_str = f'{val:.2f}' if isinstance(val, float) else str(val)
                    
                    kpi_table.append({
                        'name': name, 'value': val_str,
                        'unit': unit, 'desc': desc
                    })
                    
                    if is_card:
                        kpi_cards.append({
                            'label': name, 'value': val_str, 'unit': unit
                        })
        
        # 准备发现
        finding_list = []
        if findings:
            for f_arg in findings.get('findings', []):
                finding_list.append({
                    'severity': f_arg.get('severity', 'info'),
                    'variable': f_arg.get('variable', ''),
                    'value': f_arg.get('value', ''),
                    'message': f_arg.get('message', ''),
                    'advice': f_arg.get('advice', ''),
                })
        
        # 准备图表
        charts_b64 = {}
        if charts:
            import base64
            for name, data in charts.items():
                if isinstance(data, bytes) and len(data) > 100:
                    charts_b64[name] = base64.b64encode(data).decode()
        
        # 渲染
        try:
            from jinja2 import Template
            template = Template(REPORT_TEMPLATE)
            html = template.render(
                css=REPORT_CSS,
                title=title,
                generated_at=datetime.now().strftime('%Y-%m-%d %H:%M'),
                project_name=project_name,
                kpi_cards=kpi_cards,
                kpi_table=kpi_table,
                findings=finding_list,
                charts_png=charts_b64,
                config=config or {},
            )
            
            # PDF生成
            from weasyprint import HTML
            HTML(string=html).write_pdf(output_path)
            
            return output_path
        
        except ImportError as e:
            # fallback: save HTML
            html_path = output_path.replace('.pdf', '.html')
            with open(html_path, 'w') as f:
                f.write(html)
            return html_path
    
    @staticmethod
    def generate_html(result, kpis=None, findings=None,
                      charts=None, config=None,
                      title="管道瞬态分析报告") -> str:
        """仅生成HTML（不转PDF）"""
        return ReportGenerator.generate(
            result, kpis, findings, charts, config, title,
            output_path='/tmp/_report_temp.html'
        )
    
    @staticmethod
    def describe() -> str:
        return "Jinja2 + WeasyPrint 专业HTML→PDF报告引擎"

    @staticmethod
    def quick_json_report(kpis: dict, findings: dict = None) -> str:
        """快速JSON格式报告"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'kpi_summary': {},
            'findings': [],
        }
        for key in ['max_pressure_mpa', 'min_pressure_mpa', 'pressure_spread_mpa',
                     'pipeline_length_km', 'total_time_s']:
            if key in kpis:
                report['kpi_summary'][key] = kpis[key]
        if findings:
            report['findings'] = findings.get('findings', [])
        return json.dumps(report, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from sim.pipe import Pipe; from sim.fluid import Liquid
    from sim.solver import SinglePhaseTransientSolver, flow_inlet, pressure_outlet
    from skills.pipeline_data_processing import PipelineDataPanel
    from ai_agent.ai_agent import SimulationAdvisor
    from skills.chart_output_engine import ChartOutputEngine
    
    print("=== HTML→PDF 专业报告测试 ===")
    print()
    
    # 运行仿真
    pipe = Pipe(10000, 0.5, 0.01)
    fluid = Liquid()
    solver = SinglePhaseTransientSolver(pipe, fluid)
    result = solver.solve(15, flow_inlet(lambda t: 1.0, lambda t: 300.0),
                          pressure_outlet(lambda t: 2.0e6), 'A', P_initial=2.0e6)
    kpis = PipelineDataPanel.compute_kpis(result)
    findings = SimulationAdvisor.auto_discover(result)
    
    # 生成图表
    charts = ChartOutputEngine.generate(result, kpis, format='png', style='csdn')
    
    # 生成报告
    pdf = ReportGenerator.generate(
        result, kpis, findings, charts,
        config={'求解器': 'MOC', '管道长度': '10000m', '入口流量': '1.0 m³/s'},
        title="PipelineSim 分析报告 - 水击案例",
        project_name="测试项目",
        output_path='/tmp/pipeline_html_report.pdf'
    )
    
    sz = os.path.getsize(pdf) if os.path.exists(pdf) else 0
    print(f"  PDF报告: {sz/1024:.0f}KB ✅")
    print(f"  路径: {pdf}")
    print()
    print("✅ 报告引擎就绪!")
