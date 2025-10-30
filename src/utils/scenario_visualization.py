"""
Scenario Analysis Visualization Components.

This module provides visualization tools specifically for scenario analysis results,
including comparison charts, sensitivity analysis plots, and tornado diagrams.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any


class ScenarioVisualizer:
    """Visualization components for scenario analysis results."""
    
    def __init__(self):
        self.color_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        
    def create_scenario_comparison_chart(self, comparison_df: pd.DataFrame, 
                                       metric: str = 'Makespan (days)') -> go.Figure:
        """Create a bar chart comparing scenarios across a specific metric."""
        successful_df = comparison_df[comparison_df['Status'].isin(['OPTIMAL', 'FEASIBLE'])].copy()
        
        if successful_df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No successful scenarios to display",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=16
            )
            return fig
            
        if metric == 'Makespan (days)':
            y_values = pd.to_numeric(successful_df[metric], errors='coerce')
            y_title = 'Project Duration (Days)'
        elif metric == 'Total Cost':
            y_values = successful_df[metric].str.replace('$', '').str.replace(',', '').astype(float)
            y_title = 'Total Cost ($)'
        else:
            y_values = successful_df[metric]
            y_title = metric
            
        fig = go.Figure()
        
        colors = []
        for scenario in successful_df['Scenario']:
            if scenario == 'Baseline':
                colors.append('#2ca02c')  # Green for baseline
            else:
                colors.append('#1f77b4')  # Blue for scenarios
                
        fig.add_trace(go.Bar(
            x=successful_df['Scenario'],
            y=y_values,
            marker_color=colors,
            text=[f'{val:.1f}' if isinstance(val, (int, float)) else str(val) for val in y_values],
            textposition='auto',
            name=metric
        ))
        
        fig.update_layout(
            title=f'Scenario Comparison: {metric}',
            xaxis_title='Scenario',
            yaxis_title=y_title,
            xaxis_tickangle=-45,
            height=500,
            margin=dict(b=100)
        )
        
        return fig
    
    def create_sensitivity_tornado_chart(self, sensitivity_metrics: Dict[str, Dict]) -> go.Figure:
        """Create a tornado chart showing sensitivity of scenarios."""
        if not sensitivity_metrics:
            fig = go.Figure()
            fig.add_annotation(
                text="No sensitivity data to display",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=16
            )
            return fig
            
        scenarios = list(sensitivity_metrics.keys())
        makespan_changes = [sensitivity_metrics[s]['makespan_change_pct'] for s in scenarios]
        cost_changes = [sensitivity_metrics[s]['cost_change_pct'] for s in scenarios]
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Makespan Sensitivity (%)', 'Cost Sensitivity (%)'),
            vertical_spacing=0.15
        )
        
        colors_makespan = ['red' if x > 0 else 'green' for x in makespan_changes]
        fig.add_trace(
            go.Bar(
                x=makespan_changes,
                y=scenarios,
                orientation='h',
                marker_color=colors_makespan,
                text=[f'{val:+.1f}%' for val in makespan_changes],
                textposition='auto',
                name='Makespan Change'
            ),
            row=1, col=1
        )
        
        colors_cost = ['red' if x > 0 else 'green' for x in cost_changes]
        fig.add_trace(
            go.Bar(
                x=cost_changes,
                y=scenarios,
                orientation='h',
                marker_color=colors_cost,
                text=[f'{val:+.1f}%' for val in cost_changes],
                textposition='auto',
                name='Cost Change'
            ),
            row=2, col=1
        )
        
        for i in range(1, 3):
            fig.add_vline(x=0, line_dash="dash", line_color="black", row=i, col=1)
        
        fig.update_layout(
            title='Sensitivity Analysis: Impact vs Baseline',
            height=600,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Percentage Change from Baseline (%)", row=2, col=1)
        
        return fig
    
    def create_scenario_gantt_comparison(self, scenario_results: Dict[str, Dict], 
                                       scenarios_to_compare: List[str] = None) -> go.Figure:
        """Create side-by-side Gantt charts for scenario comparison."""
        if scenarios_to_compare is None:
            scenarios_to_compare = list(scenario_results.keys())[:3]  # Limit to 3 for readability
            
        successful_scenarios = []
        for scenario in scenarios_to_compare:
            if scenario in scenario_results and scenario_results[scenario].get('status') in ['OPTIMAL', 'FEASIBLE']:
                successful_scenarios.append(scenario)
                
        if not successful_scenarios:
            fig = go.Figure()
            fig.add_annotation(
                text="No successful scenarios to compare",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=16
            )
            return fig
            
        num_scenarios = len(successful_scenarios)
        
        fig = make_subplots(
            rows=num_scenarios, cols=1,
            subplot_titles=[f'{scenario} (Makespan: {scenario_results[scenario].get("makespan", "N/A")} days)' 
                           for scenario in successful_scenarios],
            vertical_spacing=0.15
        )
        
        for i, scenario in enumerate(successful_scenarios, 1):
            result = scenario_results[scenario]
            schedule = result.get('schedule', {})
            
            for j, (task_id, task_info) in enumerate(schedule.items()):
                fig.add_trace(
                    go.Bar(
                        x=[task_info['duration']],
                        y=[task_info['task_name']],
                        base=[task_info['start_time']],
                        orientation='h',
                        marker_color=self.color_palette[j % len(self.color_palette)],
                        text=f"{task_id}: {task_info['duration']}d",
                        textposition='inside',
                        name=task_id,
                        showlegend=(i == 1),  # Only show legend for first subplot
                        hovertemplate=(
                            f"<b>{task_info['task_name']}</b><br>"
                            f"Task ID: {task_id}<br>"
                            f"Start: Day {task_info['start_time']}<br>"
                            f"End: Day {task_info['end_time']}<br>"
                            f"Duration: {task_info['duration']} days<br>"
                            f"Labor: {task_info['labor_required']} workers"
                            "<extra></extra>"
                        )
                    ),
                    row=i, col=1
                )
        
        fig.update_layout(
            title='Gantt Chart Comparison Across Scenarios',
            height=200 * num_scenarios + 100,
            barmode='overlay'
        )
        
        for i in range(1, num_scenarios + 1):
            fig.update_xaxes(title_text="Time (Days)", row=i, col=1)
            fig.update_yaxes(title_text="Tasks", row=i, col=1)
        
        return fig
    
    def create_cost_breakdown_comparison(self, scenario_results: Dict[str, Dict]) -> go.Figure:
        """Create a stacked bar chart comparing cost breakdowns across scenarios."""
        cost_data = []
        
        for scenario_name, result in scenario_results.items():
            if result.get('status') in ['OPTIMAL', 'FEASIBLE']:
                total_cost = result.get('total_cost', 0)
                
                resource_usage = result.get('resource_usage', {})
                labor_cost = 0
                material_cost = 0
                equipment_cost = 0
                
                if resource_usage:
                    for resource_id, amount in resource_usage.items():
                        resource_lower = resource_id.lower()
                        if 'worker' in resource_lower or 'labor' in resource_lower:
                            labor_cost += amount * 100
                        elif 'material' in resource_lower or 'steel' in resource_lower or 'cement' in resource_lower or 'wood' in resource_lower:
                            material_cost += amount * 50
                        elif 'equipment' in resource_lower or 'crane' in resource_lower:
                            equipment_cost += amount * 200
                    
                    if labor_cost + material_cost + equipment_cost > 0:
                        other_cost = max(0, total_cost - labor_cost - material_cost - equipment_cost)
                    else:
                        labor_cost = total_cost * 0.6
                        material_cost = total_cost * 0.35
                        other_cost = total_cost * 0.05
                else:
                    labor_cost = total_cost * 0.6
                    material_cost = total_cost * 0.35
                    other_cost = total_cost * 0.05
                
                cost_data.append({
                    'Scenario': scenario_name,
                    'Labor Cost': labor_cost,
                    'Material Cost': material_cost,
                    'Other Cost': other_cost,
                    'Total Cost': total_cost
                })
        
        if not cost_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No cost data to display",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=16
            )
            return fig
            
        df = pd.DataFrame(cost_data)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Labor Cost',
            x=df['Scenario'],
            y=df['Labor Cost'],
            marker_color='#1f77b4',
            text=[f"${v:,.0f}" for v in df['Labor Cost']],
            textposition='inside'
        ))
        
        fig.add_trace(go.Bar(
            name='Material Cost',
            x=df['Scenario'],
            y=df['Material Cost'],
            marker_color='#ff7f0e',
            text=[f"${v:,.0f}" for v in df['Material Cost']],
            textposition='inside'
        ))
        
        fig.add_trace(go.Bar(
            name='Other Cost',
            x=df['Scenario'],
            y=df['Other Cost'],
            marker_color='#2ca02c',
            text=[f"${v:,.0f}" for v in df['Other Cost']],
            textposition='inside'
        ))
        
        fig.update_layout(
            barmode='stack',
            title='Cost Breakdown Comparison',
            xaxis_title='Scenario',
            yaxis_title='Cost ($)',
            height=500,
            showlegend=True
        )
        
        return fig
    
    def create_risk_assessment_matrix(self, scenario_results: Dict[str, Dict]) -> go.Figure:
        """Create a risk assessment matrix plotting cost vs schedule impact."""
        if not scenario_results or 'Baseline' not in scenario_results:
            fig = go.Figure()
            fig.add_annotation(
                text="Baseline data required for risk assessment",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=16
            )
            return fig
            
        baseline = scenario_results['Baseline']
        baseline_makespan = baseline.get('makespan', 1)
        baseline_cost = baseline.get('total_cost', 1)
        
        x_values = []  # Schedule impact (%)
        y_values = []  # Cost impact (%)
        labels = []
        colors = []
        
        for scenario_name, result in scenario_results.items():
            if scenario_name == 'Baseline' or result.get('status') not in ['OPTIMAL', 'FEASIBLE']:
                continue
                
            scenario_makespan = result.get('makespan', 0)
            scenario_cost = result.get('total_cost', 0)
            
            schedule_impact = ((scenario_makespan - baseline_makespan) / baseline_makespan * 100)
            cost_impact = ((scenario_cost - baseline_cost) / baseline_cost * 100)
            
            x_values.append(schedule_impact)
            y_values.append(cost_impact)
            labels.append(scenario_name)
            
            if schedule_impact >= 0 and cost_impact >= 0:
                colors.append('red')  # High risk
            elif schedule_impact < 0 and cost_impact < 0:
                colors.append('green')  # Low risk
            else:
                colors.append('orange')  # Medium risk
        
        if not x_values:
            fig = go.Figure()
            fig.add_annotation(
                text="No scenario data for risk assessment",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=16
            )
            return fig
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='markers+text',
            marker=dict(
                size=15,
                color=colors,
                line=dict(width=2, color='white')
            ),
            text=labels,
            textposition='top center',
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Schedule Impact: %{x:.1f}%<br>"
                "Cost Impact: %{y:.1f}%<br>"
                "<extra></extra>"
            )
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        
        max_x = max(x_values + [10])
        max_y = max(y_values + [10])
        min_x = min(x_values + [-10])
        min_y = min(y_values + [-10])
        
        fig.add_annotation(x=max_x*0.7, y=max_y*0.7, text="High Risk<br>(↑Cost, ↑Time)", 
                          showarrow=False, bgcolor="rgba(255,0,0,0.1)")
        fig.add_annotation(x=min_x*0.7, y=min_y*0.7, text="Low Risk<br>(↓Cost, ↓Time)", 
                          showarrow=False, bgcolor="rgba(0,255,0,0.1)")
        fig.add_annotation(x=min_x*0.7, y=max_y*0.7, text="Schedule Efficient<br>(↑Cost, ↓Time)", 
                          showarrow=False, bgcolor="rgba(255,165,0,0.1)")
        fig.add_annotation(x=max_x*0.7, y=min_y*0.7, text="Cost Efficient<br>(↓Cost, ↑Time)", 
                          showarrow=False, bgcolor="rgba(255,165,0,0.1)")
        
        fig.update_layout(
            title='Risk Assessment Matrix: Cost vs Schedule Impact',
            xaxis_title='Schedule Impact (% change from baseline)',
            yaxis_title='Cost Impact (% change from baseline)',
            height=600,
            width=800
        )
        
        return fig
    
    def create_scenario_summary_table(self, comparison_df: pd.DataFrame, 
                                    sensitivity_metrics: Dict[str, Dict]) -> go.Figure:
        """Create an interactive table summarizing all scenario results."""
        print(f"\n{'='*60}")
        print(f"CREATING SUMMARY TABLE")
        print(f"{'='*60}")
        print(f"Comparison DF shape: {comparison_df.shape}")
        print(f"Comparison DF columns: {list(comparison_df.columns)}")
        print(f"Number of rows: {len(comparison_df)}")
        if len(comparison_df) > 0:
            print(f"Scenarios in DF: {list(comparison_df['Scenario'])}")
        print(f"Sensitivity metrics keys: {list(sensitivity_metrics.keys())}")
        print(f"{'='*60}\n")
        
        enhanced_data = []
        
        for _, row in comparison_df.iterrows():
            scenario_name = row['Scenario']
            data_row = row.to_dict()
            
            if scenario_name in sensitivity_metrics:
                metrics = sensitivity_metrics[scenario_name]
                data_row['Schedule Impact (%)'] = f"{metrics['makespan_change_pct']:+.1f}%"
                data_row['Cost Impact (%)'] = f"{metrics['cost_change_pct']:+.1f}%"
                data_row['Schedule Impact (days)'] = f"{metrics['makespan_change_days']:+.1f}"
                data_row['Cost Impact ($)'] = f"${metrics['cost_change_dollars']:+,.2f}"
            else:
                data_row['Schedule Impact (%)'] = 'N/A'
                data_row['Cost Impact (%)'] = 'N/A'
                data_row['Schedule Impact (days)'] = 'N/A'
                data_row['Cost Impact ($)'] = 'N/A'
            
            enhanced_data.append(data_row)
        
        df_enhanced = pd.DataFrame(enhanced_data)
        
        print(f"Enhanced DF shape: {df_enhanced.shape}")
        print(f"Enhanced DF:\n{df_enhanced}\n")
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(df_enhanced.columns),
                fill_color='lightblue',
                align='center',
                font=dict(size=12, color='white'),
                height=40
            ),
            cells=dict(
                values=[df_enhanced[col] for col in df_enhanced.columns],
                fill_color='white',
                align='center',
                font=dict(size=11),
                height=30
            )
        )])
        
        fig.update_layout(
            title='Comprehensive Scenario Analysis Summary',
            height=400 + len(df_enhanced) * 30
        )
        
        return fig


if __name__ == "__main__":
    import pandas as pd
    
    comparison_data = {
        'Scenario': ['Baseline', 'Labor Shortage', 'Weather Delays', 'Rush Project'],
        'Status': ['OPTIMAL', 'OPTIMAL', 'FEASIBLE', 'OPTIMAL'],
        'Makespan (days)': [45, 52, 48, 38],
        'Total Cost': ['$125,000', '$140,000', '$128,000', '$145,000'],
        'Solve Time (s)': [2.1, 2.8, 3.2, 2.5],
        'Description': ['Original plan', 'Reduced labor', 'Weather delays', 'Accelerated schedule']
    }
    
    sensitivity_data = {
        'Labor Shortage': {'makespan_change_pct': 15.6, 'cost_change_pct': 12.0},
        'Weather Delays': {'makespan_change_pct': 6.7, 'cost_change_pct': 2.4},
        'Rush Project': {'makespan_change_pct': -15.6, 'cost_change_pct': 16.0}
    }
    
    df = pd.DataFrame(comparison_data)
    visualizer = ScenarioVisualizer()
    
    fig1 = visualizer.create_scenario_comparison_chart(df, 'Makespan (days)')
    fig2 = visualizer.create_sensitivity_tornado_chart(sensitivity_data)
    
    print("Scenario visualization components created successfully!")