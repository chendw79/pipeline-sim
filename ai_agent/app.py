"""
PipelineSim AI Agent — Flask API 端点

技术栈: Flask + Plotly + reportlab + NLP规则引擎
集成: ai_agent.py + pipeline_data_processing.py + dashboard
"""

import sys, os, json
from datetime import datetime
from flask import Flask, request, jsonify, send_file, render_template_string

# 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# PipelineSim核心
from sim.pipe import Pipe
from sim.fluid import Liquid
from sim.solver import SinglePhaseTransientSolver, flow_inlet, pressure_outlet

# 处理技能
from skills.pipeline_data_processing import PipelineDataPanel, PipelineProfile, PipelineHeatmap

# AI Agent
from ai_agent.ai_agent import SimulationAdvisor, PipelineReport

app = Flask(__name__)

# ─── 全局仿真结果缓存 ──────────────────────────────────
LAST_RESULT = None
LAST_CONFIG = {}

DEFAULT_PIPE = Pipe(length=10000, diameter=0.5, wall_thickness=0.01, roughness=0.05)
DEFAULT_FLUID = Liquid(name='Water')

# ─── 首页 ──────────────────────────────────────────────
INDEX_HTML = '''
<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>PipelineSim AI — 智能管道仿真</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, sans-serif;
       background: #0f0f23; color: #eee; min-height: 100vh; }
.header { background: linear-gradient(135deg, #1a1a3e, #0a0a1e);
          padding: 20px 30px; border-bottom: 1px solid #333; }
.header h1 { color: #4ecdc4; font-size: 24px; }
.header span { color: #888; font-size: 14px; }
.main { max-width: 1200px; margin: 0 auto; padding: 20px; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.card { background: #1a1a2e; border-radius: 12px; padding: 20px;
        border: 1px solid #2a2a4e; }
.card.full { grid-column: 1 / -1; }
.card h3 { color: #4ecdc4; margin-bottom: 12px; font-size: 15px; }
.input-group { margin-bottom: 12px; }
.input-group label { display: block; color: #aaa; font-size: 12px; margin-bottom: 4px; }
.input-group input, .input-group textarea, .input-group select {
    width: 100%; background: #0f0f23; border: 1px solid #333;
    color: #eee; padding: 8px 12px; border-radius: 6px; font-size: 14px; }
.input-group textarea { min-height: 60px; font-family: inherit; }
.btn { background: #4ecdc4; color: #0f0f23; border: none; padding: 8px 20px;
       border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 14px; }
.btn:hover { background: #45b7d1; }
.btn.danger { background: #e74c3c; }
.btn.warning { background: #f39c12; }
.kpi-grid { display: flex; flex-wrap: wrap; gap: 8px; }
.kpi-item { flex: 1; min-width: 120px; background: #0f0f23; border-radius: 8px;
            padding: 10px; border-left: 3px solid #4ecdc4; }
.kpi-label { font-size: 11px; color: #888; }
.kpi-value { font-size: 18px; font-weight: 700; color: #4ecdc4; }
.kpi-unit { font-size: 11px; color: #666; }
.findings { margin-top: 10px; }
.finding { padding: 8px 12px; margin: 4px 0; border-radius: 6px;
           font-size: 13px; border-left: 3px solid; }
.finding.high { background: #2a0a0a; border-color: #e74c3c; }
.finding.medium { background: #2a2000; border-color: #f39c12; }
.finding.low { background: #0a2a1a; border-color: #2ecc71; }
.finding .severity { font-size: 10px; text-transform: uppercase; }
.finding.high .severity { color: #e74c3c; }
.finding.medium .severity { color: #f39c12; }
.finding.low .severity { color: #2ecc71; }
.plot-container { background: #0f0f23; border-radius: 8px; padding: 10px;
                  margin-top: 10px; min-height: 300px; }
.plot-container iframe { width: 100%; height: 350px; border: none; }
.report-btn { margin-top: 10px; }
.loading { text-align: center; padding: 40px; color: #666; }
</style>
</head><body>
<div class="header">
    <h1>🛸 PipelineSim AI <span>— 智能管道仿真引擎</span></h1>
</div>
<div class="main">
    <div class="grid">
        <div class="card full">
            <h3>🎯 用你的话说需求</h3>
            <div class="input-group">
                <label>描述你的仿真场景</label>
                <textarea id="nlInput" placeholder='例如: "分析水击现象，入口流量从1降到0"，或"泵站急停分析"'></textarea>
            </div>
            <button class="btn" onclick="smartRun()">🚀 AI智能运行</button>
            <span style="color:#666;font-size:12px;margin-left:10px;">或手动配置 ↓</span>
        </div>

        <div class="card">
            <h3>⚙️ 管道参数</h3>
            <div class="input-group">
                <label>管道长度 (m)</label>
                <input id="pipeLen" value="10000">
            </div>
            <div class="input-group">
                <label>管径 (m)</label>
                <input id="pipeDia" value="0.5">
            </div>
            <div class="input-group">
                <label>入口流量 (m³/s)</label>
                <input id="flowRate" value="1.0">
            </div>
        </div>

        <div class="card">
            <h3>🔄 运行设置</h3>
            <div class="input-group">
                <label>求解器</label>
                <select id="solver">
                    <option value="moc">MOC 特征线法 (水击推荐)</option>
                    <option value="fvm">FVM 有限体积法</option>
                    <option value="ifdm">IFDM 隐式差分</option>
                </select>
            </div>
            <div class="input-group">
                <label>仿真时长 (s)</label>
                <input id="tMax" value="20">
            </div>
            <div class="input-group">
                <label>出口压力 (MPa)</label>
                <input id="outletP" value="2.0">
            </div>
            <button class="btn warning" onclick="manualRun()" style="margin-top:8px">▶ 手动运行</button>
        </div>

        <div class="card full" id="resultCard" style="display:none">
            <h3>📊 仿真结果</h3>
            <div id="kpiArea" class="kpi-grid"></div>
            <div id="findingsArea" class="findings"></div>
            <div class="plot-container" id="plotArea"></div>
            <div class="report-btn">
                <button class="btn" onclick="downloadPDF()">📄 下载PDF报告</button>
                <button class="btn" onclick="downloadJSON()">📋 下载JSON数据</button>
            </div>
        </div>
    </div>
</div>

<script>
function showLoading() {
    document.getElementById('resultCard').style.display = 'block';
    document.getElementById('kpiArea').innerHTML = '<div class="loading">⏳ 计算中...</div>';
    document.getElementById('findingsArea').innerHTML = '';
    document.getElementById('plotArea').innerHTML = '';
}

async function smartRun() {
    showLoading();
    const text = document.getElementById('nlInput').value || '水击分析';
    const resp = await fetch('/api/smart_run', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({query: text}),
    });
    renderResult(await resp.json());
}

async function manualRun() {
    showLoading();
    const resp = await fetch('/api/run', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            length: parseFloat(document.getElementById('pipeLen').value),
            diameter: parseFloat(document.getElementById('pipeDia').value),
            flow_rate: parseFloat(document.getElementById('flowRate').value),
            t_max: parseFloat(document.getElementById('tMax').value),
            outlet_pressure: parseFloat(document.getElementById('outletP').value),
            solver: document.getElementById('solver').value,
        }),
    });
    renderResult(await resp.json());
}

function renderResult(data) {
    const card = document.getElementById('resultCard');
    card.style.display = 'block';
    
    // KPI
    const kpis = data.kpis || {};
    let kpiHtml = '';
    const kpiDefs = [
        ['max_pressure_mpa', '最大压力', 'MPa'], ['min_pressure_mpa', '最小压力', 'MPa'],
        ['pressure_spread_mpa', '波动幅度', 'MPa'], ['inlet_pressure_mpa', '入口压力', 'MPa'],
        ['outlet_pressure_mpa', '出口压力', 'MPa'], ['pipeline_length_km', '管线长度', 'km'],
    ];
    for (const [k, label, unit] of kpiDefs) {
        if (kpis[k] !== undefined) {
            kpiHtml += '<div class="kpi-item"><div class="kpi-label">' + label +
                       '</div><div class="kpi-value">' + kpis[k].toFixed(2) +
                       ' <span class="kpi-unit">' + unit + '</span></div></div>';
        }
    }
    document.getElementById('kpiArea').innerHTML = kpiHtml;
    
    // Findings
    const findings = data.findings || [];
    let findHtml = '<h4 style="color:#888;margin:10px 0">🔍 自动分析发现</h4>';
    for (const f of findings) {
        findHtml += '<div class="finding ' + f.severity + '"><span class="severity">' +
                    f.severity + '</span> <b>' + f.variable + ':</b> ' + f.value + '<br>' +
                    f.message + '</div>';
    }
    document.getElementById('findingsArea').innerHTML = findHtml;
    
    // Plots
    document.getElementById('plotArea').innerHTML = data.plots_html || '<div class="loading">图表加载中...</div>';
}

async function downloadPDF() {
    const resp = await fetch('/api/report/pdf');
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'pipeline_report.pdf'; a.click();
}

async function downloadJSON() {
    const resp = await fetch('/api/report/json');
    const data = await resp.json();
    const jsonStr = JSON.stringify(data, null, 2);
    const blob = new Blob([jsonStr], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'pipeline_data.json'; a.click();
}
</script>
</body></html>
'''

# ─── API 端点 ─────────────────────────────────────────

@app.route('/')
def index():
    return INDEX_HTML

@app.route('/api/status')
def status():
    return jsonify({
        'version': '2.0',
        'ai_agent': True,
        'pdf_report': True,
        'solvers': ['moc', 'fvm', 'ifdm'],
    })

@app.route('/api/smart_run', methods=['POST'])
def smart_run():
    """AI自然语言→仿真"""
    data = request.get_json() or {}
    query = data.get('query', '水击分析')
    
    # AI解析
    advice = SimulationAdvisor.analyze_request(query)
    config = advice.get('config', {})
    
    return _run_simulation(config)

@app.route('/api/run', methods=['POST'])
def manual_run():
    """手动参数→仿真"""
    data = request.get_json() or {}
    
    length = data.get('length', 10000)
    diameter = data.get('diameter', 0.5)
    flow_rate = data.get('flow_rate', 1.0)
    t_max = data.get('t_max', 20)
    outlet_p = data.get('outlet_pressure', 2.0)
    solver = data.get('solver', 'moc')
    
    config = {
        'solver': solver,
        't_max': t_max,
        'bc_in': f'Q=lambda t: {flow_rate} if t < 5 else 0.0',
        'bc_out': f'P=lambda t: {outlet_p}e6',
        'P_initial': outlet_p * 1e6,
    }
    
    return _run_simulation(config)

def _run_simulation(config):
    """运行仿真并返回结果"""
    global LAST_RESULT, LAST_CONFIG
    
    try:
        pipe = DEFAULT_PIPE
        fluid = DEFAULT_FLUID
        
        t_max = config.get('t_max', 20)
        P_initial = config.get('P_initial', 2.0e6)
        
        # 创建边界条件（默认水击场景）
        bc_in = flow_inlet(Q_func=lambda t: 1.0 if t < 5 else 0.0,
                          T_func=lambda t: 300.0)
        bc_out = pressure_outlet(P_func=lambda t: 2.0e6)
        
        # 运行
        solver = SinglePhaseTransientSolver(pipe, fluid)
        result = solver.solve(t_max=t_max, inlet_bc=bc_in, outlet_bc=bc_out,
                             mode='A', P_initial=P_initial)
        
        LAST_RESULT = result
        LAST_CONFIG = config
        
        # KPI
        kpis = PipelineDataPanel.compute_kpis(result)
        
        # AI发现
        findings = SimulationAdvisor.auto_discover(result)
        
        # 图表HTML
        profile_fig = PipelineProfile.along_pipe(result, 'P')
        heatmap_fig = PipelineHeatmap.spacetime_heatmap(result, 'P')
        
        plots_html = profile_fig.to_html(full_html=False, include_plotlyjs='cdn') + \
                     heatmap_fig.to_html(full_html=False, include_plotlyjs=False)
        
        return jsonify({
            'status': 'ok',
            'kpis': kpis,
            'findings': findings.get('findings', []),
            'summary': findings.get('summary', ''),
            'plots_html': plots_html,
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/report/pdf')
def download_pdf():
    """下载PDF报告"""
    global LAST_RESULT
    if LAST_RESULT is None:
        return jsonify({'error': '请先运行仿真'})
    
    kpis = PipelineDataPanel.compute_kpis(LAST_RESULT)
    findings = SimulationAdvisor.auto_discover(LAST_RESULT)
    
    pdf_path = PipelineReport.generate(LAST_RESULT, kpis, findings)
    return send_file(pdf_path, as_attachment=True,
                    download_name=f'pipeline_report_{datetime.now().strftime("%Y%m%d_%H%M")}.pdf')

@app.route('/api/report/json')
def download_json():
    """下载JSON报告"""
    global LAST_RESULT
    if LAST_RESULT is None:
        return jsonify({'error': '请先运行仿真'})
    
    kpis = PipelineDataPanel.compute_kpis(LAST_RESULT)
    findings = SimulationAdvisor.auto_discover(LAST_RESULT)
    
    return jsonify({
        'kpis': kpis,
        'findings': findings.get('findings', []),
        'summary': findings.get('summary', ''),
        'recommendations': [f.get('advice', '') for f in findings.get('findings', [])],
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8768))
    print(f"🚀 PipelineSim AI Agent — http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
