"""
PipelineSim AI Agent — 智能仿真引擎
====================================
Jobs+Musk 设计理念:
1. 自然语言输入 → 自动配置仿真 (Elon: 减少80%手动操作)
2. 自动发现关键洞察 → 智能标注峰谷值 (Steve: 好设计告诉你看什么)
3. 一键报告生成 → PDF输出 (Jobs: 看不见的地方也要完美)
"""

import os, json, io, re
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════
# AI仿真助手 — 自然语言→仿真配置
# ═══════════════════════════════════════════════

class SimulationAdvisor:
    """仿真顾问 — 规则引擎版AI助手"""
    
    # 场景模板库
    SCENARIOS = {
        'water_hammer': {
            'name': '水击分析',
            'description': '阀门快速关闭/开启引起的水击压力波',
            'keywords': ['水击', '水锤', 'water hammer', '快关', '快开', 'sudden', 'valve'],
            'setup': {
                'solver': 'moc',
                't_max': 20,
                'bc_in_type': 'flow',
                'bc_in': 'Q=lambda t: 1.0 if t < 5 else 0.0',
                'bc_out_type': 'pressure',
                'bc_out': 'P=lambda t: 2.0e6',
                'P_initial': 2.0e6,
            },
            'description_template': '阀门操作引起的水击压力波传播分析。初始流量1.0m³/s，t=5s时阀门关闭。',
        },
        'pump_trip': {
            'name': '泵站停泵分析',
            'description': '泵站突然停机引起的瞬态压力变化',
            'keywords': ['泵', '停泵', 'pump', 'trip', '急停', 'shutdown'],
            'setup': {
                'solver': 'fvm',
                't_max': 30,
                'bc_in_type': 'flow',
                'bc_in': 'Q=lambda t: 1.0 if t < 2 else 0.0',
                'bc_out_type': 'pressure',
                'bc_out': 'P=lambda t: 1.0e6',
                'P_initial': 1.5e6,
            },
            'description_template': '泵站突然停机引起的管道瞬态分析。初始流量1.0m³/s，t=2s时停泵。',
        },
        'steady_state': {
            'name': '稳态分析',
            'description': '管道稳态运行压力/温度/流速分析',
            'keywords': ['稳态', 'steady', '正常运行', 'normal', '常规'],
            'setup': {
                'solver': 'moc',
                't_max': 5,
                'bc_in_type': 'flow',
                'bc_in': 'Q=lambda t: 1.0',
                'bc_out_type': 'pressure',
                'bc_out': 'P=lambda t: 2.0e6',
                'P_initial': 2.0e6,
            },
            'description_template': '管道稳态运行分析。稳定流量1.0m³/s。',
        },
        'batch_sweep': {
            'name': '批次扫描分析',
            'description': '不同操作参数下的系统响应对比',
            'keywords': ['批次', 'batch', '对比', 'compare', '不同', '参数扫描', 'sweep'],
            'setup': {
                'solver': 'moc',
                't_max': 30,
                'bc_in_type': 'flow',
                'bc_in': 'Q=lambda t: 1.0 if t < 10 else 0.5',
                'bc_out_type': 'pressure',
                'bc_out': 'P=lambda t: 2.0e6',
                'P_initial': 2.0e6,
            },
            'description_template': '批次输送瞬态分析。流量从1.0m³/s降至0.5m³/s。',
        },
    }
    
    SOLVER_ADVICE = {
        'moc': {
            'best_for': '水击/阀门操作/有界问题',
            'pros': '精确捕捉波前, 计算快',
            'cons': 'Courant约束, 需等距网格',
        },
        'fvm': {
            'best_for': '复杂几何/间断流/非等温',
            'pros': '守恒性好, 处理间断',
            'cons': '比MOC慢, 需黎曼求解器',
        },
        'ifdm': {
            'best_for': '慢瞬态/长时间/大系统',
            'pros': '无条件稳定, 大时间步长',
            'cons': '耗散性强, 不精确捕捉波前',
        },
    }
    
    @staticmethod
    def analyze_request(nl_input: str) -> dict:
        """
        自然语言→仿真配置
        
        Args:
            nl_input: 用户输入，如"分析管道水击，入口流量变化0.3"
        
        Returns:
            匹配的场景配置和参数
        """
        desc_lower = nl_input.lower()
        
        # 1. 匹配场景
        matched_scenario = None
        best_score = 0
        for sid, scenario in SimulationAdvisor.SCENARIOS.items():
            score = sum(1 for kw in scenario['keywords'] if kw in desc_lower)
            if score > best_score:
                best_score = score
                matched_scenario = sid
        
        # 2. 提取参数
        params = SimulationAdvisor._extract_params(nl_input)
        
        # 3. 构建配置
        config = {}
        if matched_scenario:
            config = SimulationAdvisor.SCENARIOS[matched_scenario]['setup'].copy()
        
        # 覆盖用户指定的参数
        config.update(params)
        
        # 求解器建议
        solver_advice = SimulationAdvisor.SOLVER_ADVICE.get(
            config.get('solver', 'moc'), {}
        )
        
        return {
            'matched_scenario': matched_scenario,
            'description': SimulationAdvisor.SCENARIOS[matched_scenario]['description_template'] if matched_scenario else '自定义仿真',
            'config': config,
            'solver_advice': solver_advice,
            'confidence': best_score / max(len(SimulationAdvisor.SCENARIOS[matched_scenario]['keywords']), 1) if matched_scenario else 0,
            'extracted_params': params,
        }
    
    @staticmethod
    def _extract_params(text: str) -> dict:
        """从文本中提取数值参数"""
        params = {}
        
        # 流量
        flow_match = re.findall(r'流量[约]?(\d+\.?\d*)', text)
        if flow_match:
            params['initial_flow'] = float(flow_match[0])
        
        # 压力 (MPa)
        pressure_match = re.findall(r'压力[约]?(\d+\.?\d*)', text)
        if pressure_match:
            params['P_setting'] = float(pressure_match[0]) * 1e6
        
        # 时间
        time_match = re.findall(r'(\d+\.?\d*)[秒s]', text)
        if time_match:
            params['t_setting'] = float(time_match[0])
        
        return params

    @staticmethod
    def auto_discover(result) -> dict:
        """
        自动发现关键洞察
        
        Args:
            result: TransientResult
        
        Returns:
            关键发现列表
        """
        P = getattr(result, 'P', None)
        t_arr = getattr(result, 't', None)
        x_arr = getattr(result, 'x', None)
        
        if P is None:
            return {}
        
        findings = []
        
        # 1. 最大压力 + 位置 + 时间
        max_idx = np.unravel_index(np.argmax(P), P.shape)
        max_p = P[max_idx] / 1e6
        max_t = t_arr[max_idx[0]] if t_arr is not None else 0
        max_x = x_arr[max_idx[1]] / 1000 if x_arr is not None else 0
        
        findings.append({
            'type': 'danger' if max_p > 5.0 else 'warning',
            'variable': '压力',
            'value': f'{max_p:.2f} MPa',
            'location': f'x={max_x:.1f}km',
            'time': f't={max_t:.1f}s',
            'message': f'最大压力 {max_p:.2f}MPa，出现在 {max_x:.1f}km 处，时刻 {max_t:.1f}s',
            'advice': '超过管道设计压力！建议：\n  - 延长阀门关闭时间\n  - 安装水击泄压阀\n  - 检查管道压力等级' if max_p > 5.0 else '需关注压力峰值',
            'severity': 'high' if max_p > 5.0 else 'medium',
        })
        
        # 2. 最小压力
        min_p = float(np.min(P) / 1e6)
        if min_p < 0:
            findings.append({
                'type': 'danger',
                'variable': '压力',
                'value': f'{min_p:.2f} MPa',
                'message': f'出现负压 {min_p:.2f}MPa — 有汽蚀/空化风险！',
                'advice': '负压可能导致管道失稳甚至坍塌。建议安装真空破坏阀或减缓操作速度。',
                'severity': 'high',
            })
        elif min_p < 0.5:
            findings.append({
                'type': 'warning',
                'variable': '压力',
                'value': f'{min_p:.2f} MPa',
                'message': f'低压区 {min_p:.2f}MPa — 接近汽化压力',
                'advice': '建议检查管网是否有可能产生汽蚀的位置。',
                'severity': 'medium',
            })
        
        # 3. 波动幅度
        Pspread = float((np.max(P) - np.min(P)) / 1e6)
        if Pspread > 3.0:
            findings.append({
                'type': 'warning',
                'variable': '波动幅度',
                'value': f'{Pspread:.2f} MPa',
                'message': f'压力波动幅度 {Pspread:.2f}MPa — 系统承受较大交变载荷',
                'advice': '大幅度波动可能导致疲劳失效。建议采用多阶段操作或增加缓冲装置。',
                'severity': 'medium',
            })
        
        # 4. 波速
        if t_arr is not None and len(P) > 1:
            wave_speed = 1200  # 默认
            findings.append({
                'type': 'info',
                'variable': '压力波速',
                'value': f'{wave_speed:.0f} m/s',
                'message': f'压力波传播速度约{wave_speed:.0f}m/s',
                'advice': f'波速由管壁弹性模量和流体压缩性决定。如需精确值，请提供管道材料参数。',
                'severity': 'low',
            })
        
        # 5. 稳态判断
        if t_arr is not None and len(P) > 10:
            last_p = P[-1, :] / 1e6
            first_p = P[0, :] / 1e6
            if np.max(np.abs(last_p - first_p)) < 0.1:
                findings.append({
                    'type': 'success',
                    'variable': '系统状态',
                    'value': '已稳定',
                    'message': '系统在仿真结束时已达到稳态',
                    'advice': '参数设置合理，系统收敛正常。',
                    'severity': 'low',
                })
        
        return {
            'findings': findings,
            'summary': f'分析完成。共发现 {len(findings)} 个关键点，其中'
                       f' {sum(1 for f in findings if f["severity"]=="high")} 项高风险。',
        }
    
    @staticmethod
    def evaluate_param_change(result_original, result_modified) -> dict:
        """
        对比两次结果，生成"如果这样会怎样"的评估
        
        Args:
            result_original: 原始结果
            result_modified: 修改后结果
        
        Returns:
            对比报告
        """
        Po = getattr(result_original, 'P', None)
        Pm = getattr(result_modified, 'P', None)
        
        if Po is None or Pm is None:
            return {}
        
        max_po = np.max(Po) / 1e6
        max_pm = np.max(Pm) / 1e6
        
        reduction = (max_po - max_pm) / max_po * 100 if max_po > 0 else 0
        
        return {
            'original_max_pressure': f'{max_po:.2f} MPa',
            'modified_max_pressure': f'{max_pm:.2f} MPa',
            'peak_reduction_pct': f'{reduction:.1f}%',
            'effectiveness': '✅ 有效降低' if reduction > 10 else '⚠️ 效果有限' if reduction > 0 else '❌ 无改善',
            'advice': '参数调整有效，建议进一步优化关闭曲线。' if reduction > 10 else '建议尝试其他参数组合。',
        }


# ═══════════════════════════════════════════════
# PDF报告自动生成
# ═══════════════════════════════════════════════

class PipelineReport:
    """管道仿真报告 — 一键PDF生成"""
    
    @staticmethod
    def generate(result, kpis: dict = None, findings: dict = None,
                  output_path: str = None, title: str = '管道瞬态分析报告') -> str:
        """
        生成专业PDF报告
        
        Args:
            result: TransientResult
            kpis: 来自PipelineDataPanel.compute_kpis
            findings: 来自SimulationAdvisor.auto_discover
            output_path: 输出PDF路径
            title: 报告标题
        """
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Table, TableStyle, Image, PageBreak)
        from reportlab.lib.units import mm, cm
        import io
        
        if output_path is None:
            output_path = f'/tmp/pipeline_report_{int(datetime.now().timestamp())}.pdf'
        
        # 设置文档
        doc = SimpleDocTemplate(output_path, pagesize=A4,
                                rightMargin=20*mm, leftMargin=20*mm,
                                topMargin=20*mm, bottomMargin=20*mm)
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('CustomTitle', parent=styles['Title'],
                                     fontSize=22, spaceAfter=20,
                                     textColor=colors.HexColor('#1a1a2e'))
        heading1 = ParagraphStyle('H1', parent=styles['Heading1'],
                                  fontSize=16, spaceBefore=15, spaceAfter=8,
                                  textColor=colors.HexColor('#2c3e50'))
        heading2 = ParagraphStyle('H2', parent=styles['Heading2'],
                                  fontSize=13, spaceBefore=10, spaceAfter=6,
                                  textColor=colors.HexColor('#34495e'))
        normal = ParagraphStyle('Norm', parent=styles['Normal'],
                                fontSize=10, spaceAfter=4, leading=14)
        
        elements = []
        
        # 标题页
        elements.append(Paragraph(title, title_style))
        elements.append(Paragraph(
            f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M")}', normal))
        elements.append(Spacer(1, 10*mm))
        
        # KPI摘要表
        if kpis:
            elements.append(Paragraph('KPI 摘要', heading1))
            kpi_data = [['指标', '数值', '单位']]
            key_map = {
                'max_pressure_mpa': '最大压力', 'min_pressure_mpa': '最小压力',
                'pressure_spread_mpa': '波动幅度', 'inlet_pressure_mpa': '入口压力',
                'outlet_pressure_mpa': '出口压力', 'pipeline_length_km': '管线长度',
                'total_time_s': '仿真时长', 'max_temp_c': '最高温度',
            }
            for key, label in key_map.items():
                if key in kpis:
                    unit = 'MPa' if 'pressure' in key or 'spread' in key else \
                           'km' if 'length' in key else \
                           '°C' if 'temp' in key else 's'
                    v = kpis[key]
                    if isinstance(v, float):
                        kpi_data.append([label, f'{v:.2f}', unit])
            
            kpi_table = Table(kpi_data, colWidths=[80, 80, 50])
            kpi_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
            ]))
            elements.append(kpi_table)
            elements.append(Spacer(1, 5*mm))
        
        # 自动发现
        if findings:
            elements.append(Paragraph('自动分析发现', heading1))
            for f in findings.get('findings', []):
                severity_colors = {'high': colors.HexColor('#e74c3c'),
                                   'medium': colors.HexColor('#f39c12'),
                                   'low': colors.HexColor('#2ecc71')}
                fc = severity_colors.get(f.get('severity', 'low'), colors.black)
                
                elements.append(Paragraph(
                    f'<font color="{fc.hexval()}">●</font> '
                    f'<b>{f.get("variable", "")}: {f.get("value", "")}</b>', normal))
                elements.append(Paragraph(f.get('message', ''), normal))
                if f.get('advice'):
                    elements.append(Paragraph(
                        f'<i>建议: {f.get("advice", "")}</i>', normal))
                elements.append(Spacer(1, 3*mm))
        
        # 生成图表
        try:
            import plotly.io as pio
            import plotly.graph_objects as go
            
            elements.append(Paragraph('仿真结果图表', heading1))
            
            # 沿程图
            from skills.pipeline_data_processing import PipelineProfile
            fig = PipelineProfile.along_pipe(result, 'P')
            img_bytes = pio.to_image(fig, format='png', width=700, height=350, scale=1.5)
            elements.append(Image(io.BytesIO(img_bytes), width=160*mm, height=80*mm))
            elements.append(Spacer(1, 5*mm))
            
            # 热力图
            from skills.pipeline_data_processing import PipelineHeatmap
            fig = PipelineHeatmap.spacetime_heatmap(result, 'P')
            img_bytes = pio.to_image(fig, format='png', width=700, height=350, scale=1.5)
            elements.append(Image(io.BytesIO(img_bytes), width=160*mm, height=80*mm))
            
        except Exception as e:
            elements.append(Paragraph(f'<i>图表生成: {e}</i>', normal))
        
        # 构建PDF
        doc.build(elements)
        return output_path

    @staticmethod
    def simple_json_report(result, kpis: dict = None, findings: dict = None) -> str:
        """生成JSON格式的简短报告"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'simulation': {},
            'kpi_summary': {},
            'findings': [],
        }
        
        if kpis:
            for key in ['max_pressure_mpa', 'min_pressure_mpa', 'pressure_spread_mpa',
                        'pipeline_length_km', 'total_time_s']:
                if key in kpis:
                    report['kpi_summary'][key] = kpis[key]
        
        if findings:
            report['findings'] = findings.get('findings', [])
            report['summary'] = findings.get('summary', '')
        
        return json.dumps(report, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════
# 导出
# ═══════════════════════════════════════════════

__all__ = ['SimulationAdvisor', 'PipelineReport']


if __name__ == '__main__':
    print("=" * 55)
    print("  PipelineSim AI Agent + 智能报告引擎")
    print("  ├─ SimulationAdvisor: 自然语言→仿真配置")
    print("  │   ├─ analyze_request   — 语义解析")
    print("  │   ├─ auto_discover     — 自动发现洞察")
    print('  │   └─ evaluate_param_change — 如果这样 对比')
    print("  ├─ PipelineReport: PDF报告生成")
    print("  │   ├─ generate          — 一键PDF")
    print("  │   └─ simple_json_report— JSON报告")
    print("=" * 55)
    
    # 测试AI助手
    print("\n测试自然语言解析...")
    tests = [
        "分析水击，关闭阀门",
        "泵站急停分析",
        "稳态运行看看",
        "帮我做批次输送模拟",
    ]
    for t in tests:
        adv = SimulationAdvisor.analyze_request(t)
        scenario = adv['matched_scenario']
        solver = adv['config'].get('solver', '?')
        print(f"  '{t}' → {scenario} (求解器={solver}) confidence={adv['confidence']:.0%}")
    
    print("\n测试自动发现（使用模拟数据）...")
    from sim.pipe import Pipe; from sim.fluid import Liquid
    from sim.solver import SinglePhaseTransientSolver, flow_inlet, pressure_outlet
    solver = SinglePhaseTransientSolver(Pipe(10000, 0.5, 0.01), Liquid())
    result = solver.solve(15, flow_inlet(lambda t: 1.0, lambda t: 300.0),
                          pressure_outlet(lambda t: 2.0e6), 'A', P_initial=2.0e6)
    
    findings = SimulationAdvisor.auto_discover(result)
    for f in findings.get('findings', []):
        print(f"  [{f['severity']}] {f['variable']}: {f['value']} — {f['message'][:50]}")
    
    print("\n测试PDF报告...")
    from skills.pipeline_data_processing import PipelineDataPanel
    kpis = PipelineDataPanel.compute_kpis(result)
    pdf = PipelineReport.generate(result, kpis, findings,
                                   output_path='/tmp/pipeline_report_test.pdf')
    import os
    print(f"  PDF: {os.path.getsize(pdf)/1024:.0f}KB ✅")
    
    print("\n✅ AI Agent + 报告引擎就绪!")