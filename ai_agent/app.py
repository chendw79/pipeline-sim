"""PipelineSim AI Agent — Flask API 端点 (v3.0 集成版)"""
import sys, os, json, io, base64
from datetime import datetime
from flask import Flask, request, jsonify, send_file, render_template_string

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from sim.pipe import Pipe
from sim.fluid import Liquid
from sim.solver import SinglePhaseTransientSolver, flow_inlet, pressure_outlet
from skills.pipeline_data_processing import PipelineDataPanel, PipelineProfile, PipelineHeatmap
from ai_agent.ai_agent import SimulationAdvisor, PipelineReport

# 新导入
from skills.data_input_pipeline import (
    WordParser, ExcelParser, PDFParser, parse_document,
    config_to_setup, SimulationConfig, PipeConfig, BatchConfig
)
from skills.chart_output_engine import ChartOutputEngine, MPLChartEngine, StyleTemplate
from skills.report_generator import ReportGenerator

# 非牛顿流体
from sim.fluids import PowerLawFluid, BinghamFluid, HerschelBulkleyFluid, describe_fluid
# 阀门
from sim.valves import ValveCv, PIDController, ControlValve, step_valve_schedule
# 泄漏
from sim.leaks import PipeLeak, leak_corrected_solve, detect_leak_from_data
# T-P耦合
from sim.thermal_coupling import CoupledThermalSolver, ThermallyCoupledFluid

app = Flask(__name__)
LAST_RESULT = None
LAST_CHARTS = {}
DEFAULT_PIPE = Pipe(length=10000, diameter=0.5, wall_thickness=0.01, roughness=0.05)
DEFAULT_FLUID = Liquid(name='Water')

HOME = '''<!DOCTYPE html><html><head><meta charset=utf-8>
<title>PipelineSim AI v3.0</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,sans-serif;background:#0f0f23;color:#eee;line-height:1.5}
.header{background:linear-gradient(135deg,#1a1a3e,#0a0a1e);padding:20px;border-bottom:1px solid #333}
.header h1{color:#4ecdc4;font-size:22px}
.header span{font-size:13px;color:#888}
.main{max-width:1200px;margin:0 auto;padding:16px}
.card{background:#1a1a2e;border-radius:10px;padding:16px;margin-bottom:12px;border:1px solid #2a2a4e}
.card h3{color:#4ecdc4;margin-bottom:10px;font-size:14px}
.flex{display:flex;gap:12px;flex-wrap:wrap}
.btn{background:#4ecdc4;color:#0f0f23;border:none;padding:7px 16px;border-radius:6px;cursor:pointer;font-weight:600;font-size:13px;margin:3px}
.btn:hover{background:#45b7d1}
.btn.sec{background:#34495e;color:#eee}
.btn.sec:hover{background:#2c3e50}
textarea,input,select{width:100%;background:#0f0f23;border:1px solid #444;color:#eee;padding:8px;border-radius:6px;margin-bottom:8px;font-size:13px}
textarea{min-height:50px}
.kpi-grid{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}
.kpi-item{flex:1;min-width:100px;background:#0f0f23;border-radius:6px;padding:8px;border-left:3px solid #4ecdc4}
.kpi-label{font-size:10px;color:#888}
.kpi-value{font-size:16px;font-weight:700;color:#4ecdc4}
.finding{padding:6px 10px;margin:3px 0;border-radius:4px;font-size:12px;border-left:3px solid}
.finding.high{background:#2a0a0a;border-color:#e74c3c}
.finding.medium{background:#2a2000;border-color:#f39c12}
.finding.low{background:#0a2a1a;border-color:#2ecc71}
.tag{display:inline-block;padding:2px 8px;border-radius:10px;font-size:10px;margin:2px;background:#2a2a4e}
.tag.new{background:#1a4a2e;color:#4ecdc4}
</style></head><body>
<div class=header><h1>PipelineSim AI v3.0 <span>— 管道瞬态仿真智能引擎</span></h1></div>
<div class=main>
<div class=card><h3> 用你的话说需求</h3>
<textarea id=nl placeholder='例如: "分析水击"、"热油管线温度耦合"、"非牛顿流体泥浆"、"泄漏检测"、或"PID压力控制"'></textarea>
<button class=btn onclick=smartRun()> AI 智能运行</button>
<span class=tag.new>新! 泄漏/PID/阀门/非牛顿/T-P耦合</span>
</div>

<div class=card><h3> 图表输出风格</h3>
<select id=style><option value=csdn>CSDN文章配图 (白底)</option><option value=consulting>咨询报告 (McKinsey风)</option><option value=tech_dark>科技暗色 (大屏风)</option></select>
<div class=flex>
<button class="btn sec" onclick=downloadChart('png')> 下载CSDN静态图</button>
<button class="btn sec" onclick=downloadChart('pyecharts')> 下载ECharts交互图</button>
</div></div>

<div id=result style=display:none>
<div class=card><h3> 仿真结果</h3>
<div id=kpiArea class=kpi-grid></div>
<div id=findingsArea></div></div>
<div class=card><h3> 图表</h3>
<div id=plotArea></div>
<div class=flex>
<button class=btn onclick=downloadPDF()> PDF报告(带图)</button>
<button class=btn onclick=downloadReport('html')> HTML报告</button>
<button class="btn sec" onclick=downloadReport('json')> JSON数据</button>
</div></div></div>

<script>
async function smartRun(){
document.getElementById('result').style.display='block';
document.getElementById('kpiArea').innerHTML='<div style="padding:20px;color:#666">计算中...</div>';
const q=document.getElementById('nl').value||'水击分析';
const s=document.getElementById('style').value;
const r=await fetch('/api/smart_run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q,style:s})});
const d=await r.json();renderResult(d)}
function renderResult(d){
const k=d.kpis||{};let kh='';
[['max_pressure_mpa','最大压力','MPa'],['min_pressure_mpa','最小压力','MPa'],['pressure_spread_mpa','波动','MPa'],['pipeline_length_km','管长','km']].forEach(([k2,l,u])=>{
if(k[k2]!==undefined)kh+='<div class=kpi-item><div class=kpi-label>'+l+'</div><div class=kpi-value>'+k[k2].toFixed(2)+' <span style=font-size:10px;color:#666>'+u+'</span></div></div>'});
document.getElementById('kpiArea').innerHTML=kh;
let fh='<h4 style=color:#888;margin:8px 0>自动发现 ('+(d.findings||[]).length+'项)</h4>';
(d.findings||[]).forEach(f=>{fh+='<div class="finding '+f.severity+'"><b>'+f.variable+':</b> '+f.value+'<br>'+f.message+'</div>'});
document.getElementById('findingsArea').innerHTML=fh;
document.getElementById('plotArea').innerHTML=d.plots_html||''}
async function downloadChart(t){document.getElementById('nl').value='请先运行仿真，图表将在结果栏显示'}
async function downloadPDF(){const r=await fetch('/api/report/pdf');const b=await r.blob();const u=URL.createObjectURL(b);const a=document.createElement('a');a.href=u;a.download='pipeline_report.pdf';a.click()}
async function downloadReport(t){const r=await fetch('/api/report/'+t);const d=await r.json();const b=new Blob([JSON.stringify(d,null,2)],{type:'text/plain'});const u=URL.createObjectURL(b);const a=document.createElement('a');a.href=u;a.download='report.'+(t=='json'?'json':'txt');a.click()}
</script></body></html>'''

@app.route('/') 
def index(): return HOME

@app.route('/api/status')
def api_status():
    return jsonify({
        'version': '3.0',
        'features': {
            'ai_agent': True, 'pdf_report': True, 'data_input': True,
            'chart_output': ['matplotlib', 'seaborn', 'pyecharts', 'plotly'],
            'models': ['non_newtonian', 'valve_cv', 'pid_control',
                      'control_valve', 'leak_detection', 'tp_coupling'],
            'styles': ['csdn', 'consulting', 'tech_dark'],
            'input_formats': ['docx', 'xlsx', 'pdf'],
        },
        'solvers': ['moc', 'fvm', 'ifdm'],
    })

@app.route('/api/smart_run', methods=['POST'])
def api_smart_run():
    data = request.get_json() or {}
    query = data.get('query', '水击分析')
    style = data.get('style', 'csdn')
    advice = SimulationAdvisor.analyze_request(query)
    return _run_sim(advice.get('config', {}), style)

def _run_sim(config, style='csdn'):
    global LAST_RESULT, LAST_CHARTS
    try:
        t_max = config.get('t_max', 20)
        P_initial = config.get('P_initial', 2.0e6)
        solver = SinglePhaseTransientSolver(DEFAULT_PIPE, DEFAULT_FLUID)
        bc_in = flow_inlet(Q_func=lambda t: 1.0 if t < 5 else 0.0, T_func=lambda t: 300.0)
        bc_out = pressure_outlet(P_func=lambda t: 2.0e6)
        result = solver.solve(t_max=t_max, inlet_bc=bc_in, outlet_bc=bc_out,
                             mode='A', P_initial=P_initial)
        LAST_RESULT = result
        
        kpis = PipelineDataPanel.compute_kpis(result)
        findings = SimulationAdvisor.auto_discover(result)
        
        # 多引擎图表
        LAST_CHARTS = ChartOutputEngine.generate(result, kpis, format='png', style=style)
        
        # Plotly交互图(原有)
        profile_fig = PipelineProfile.along_pipe(result, 'P')
        heatmap_fig = PipelineHeatmap.spacetime_heatmap(result, 'P')
        plots = profile_fig.to_html(full_html=False, include_plotlyjs='cdn') + \
                heatmap_fig.to_html(full_html=False, include_plotlyjs=False)
        
        return jsonify({
            'status': 'ok', 'kpis': kpis,
            'findings': findings.get('findings', []),
            'summary': findings.get('summary', ''),
            'plots_html': plots,
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/report/pdf')
def api_pdf():
    global LAST_RESULT, LAST_CHARTS
    if LAST_RESULT is None: return jsonify({'error': '请先运行仿真'})
    kpis = PipelineDataPanel.compute_kpis(LAST_RESULT)
    findings = SimulationAdvisor.auto_discover(LAST_RESULT)
    pdf = ReportGenerator.generate(
        LAST_RESULT, kpis, findings, LAST_CHARTS,
        {'求解器': 'MOC', '管道': '10000m×0.5m'},
        'PipelineSim 分析报告',
        output_path=f'/tmp/pipeline_v3_{int(datetime.now().timestamp())}.pdf'
    )
    return send_file(pdf, as_attachment=True,
                    download_name=f'report_{datetime.now().strftime("%Y%m%d")}.pdf')

@app.route('/api/report/json')
def api_json():
    if LAST_RESULT is None: return jsonify({'error': '请先运行仿真'})
    kpis = PipelineDataPanel.compute_kpis(LAST_RESULT)
    findings = SimulationAdvisor.auto_discover(LAST_RESULT)
    return jsonify({'kpis': kpis, 'findings': findings.get('findings', [])})

@app.route('/api/report/html')
def api_html():
    if LAST_RESULT is None: return jsonify({'error': '请先运行仿真'})
    kpis = PipelineDataPanel.compute_kpis(LAST_RESULT)
    findings = SimulationAdvisor.auto_discover(LAST_RESULT)
    html = ReportGenerator.generate_html(LAST_RESULT, kpis, findings, LAST_CHARTS)
    with open(html) as f:
        return f.read(), 200, {'Content-Type': 'text/html'}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8768))
    print(f"PipelineSim AI v3.0 — http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
