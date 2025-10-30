"""
Visualization Components for Construction Scheduling System.

This module provides Plotly-based visualizations including Gantt charts,
resource allocation graphs, and performance dashboards.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import streamlit as st


class ScheduleVisualizer:
    """Creates visualizations for construction scheduling results."""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
        
    def create_gantt_chart(self, schedule_data: Dict, title: str = "Construction Schedule") -> go.Figure:
        """Create an interactive Gantt chart from schedule data."""
        
        gantt_data = []
        
        for task_id, task_info in schedule_data.items():
            start_day = task_info['start_time']
            end_day = task_info['end_time']
            duration = task_info.get('duration', end_day - start_day)
            
            gantt_data.append({
                'Task': task_info.get('task_name', task_id),
                'Start': start_day,
                'Finish': end_day,
                'Duration': duration,
                'Labor': task_info.get('labor_required', 0),
                'Task_ID': task_id
            })
            
        df = pd.DataFrame(gantt_data)
        df = df.sort_values('Start')
        
        fig = go.Figure()
        
        for idx, row in df.iterrows():
            fig.add_trace(go.Bar(
                x=[row['Duration']],
                y=[row['Task']],
                orientation='h',
                base=row['Start'],
                name=row['Task'],
                marker=dict(
                    color=row['Duration'],
                    colorscale='viridis',
                    showscale=idx == 0,
                    colorbar=dict(title="Duration (days)") if idx == 0 else None
                ),
                hovertemplate=(
                    f"<b>{row['Task']}</b><br>" +
                    f"Task ID: {row['Task_ID']}<br>" +
                    f"Start: Day {row['Start']}<br>" +
                    f"End: Day {row['Finish']}<br>" +
                    f"Duration: {row['Duration']} days<br>" +
                    f"Labor: {row['Labor']}<br>" +
                    "<extra></extra>"
                ),
                showlegend=False
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Project Day",
            yaxis_title="Tasks",
            height=max(400, len(gantt_data) * 40),
            barmode='overlay',
            hovermode='closest',
            yaxis=dict(autorange="reversed")
        )
            
        return fig
    
    def create_resource_utilization_chart(self, schedule_data: Dict, resources_data: Dict = None) -> go.Figure:
        """Create resource utilization timeline chart."""
        
        max_time = max(task_info['end_time'] for task_info in schedule_data.values())
        time_points = list(range(max_time + 1))
        
        labor_usage = [0] * (max_time + 1)
        
        for task_info in schedule_data.values():
            start = task_info['start_time']
            end = task_info['end_time']
            
            labor = 0
            if 'labor_required' in task_info:
                labor = task_info['labor_required']
            elif 'resource_assignments' in task_info:
                for resource_id, amount in task_info['resource_assignments'].items():
                    if 'worker' in resource_id.lower() or 'labor' in resource_id.lower():
                        labor += amount
            
            for t in range(start, end):
                if t < len(labor_usage):
                    labor_usage[t] += labor
                    
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=time_points,
            y=labor_usage,
            mode='lines+markers',
            name='Labor Usage',
            line=dict(color='blue', width=2),
            fill='tozeroy',
            fillcolor='rgba(0,100,255,0.2)'
        ))
        
        if resources_data:
            labor_capacity = None
            
            for resource_id, resource_info in resources_data.items():
                if isinstance(resource_info, (int, float)):
                    continue
                elif isinstance(resource_info, dict):
                    if resource_info.get('resource_type') == 'labor' or 'labor' in resource_id.lower():
                        labor_capacity = resource_info.get('total_capacity') or resource_info.get('capacity')
                        break
                else:
                    if hasattr(resource_info, 'resource_type') and resource_info.resource_type == 'labor':
                        labor_capacity = resource_info.capacity
                        break
                    elif 'labor' in resource_id.lower():
                        if hasattr(resource_info, 'capacity'):
                            labor_capacity = resource_info.capacity
                            break
                            
            if labor_capacity is None and labor_usage:
                labor_capacity = max(labor_usage) + 2  # Add some buffer
                
            if labor_capacity:
                fig.add_hline(y=labor_capacity, 
                             line_dash="dash", 
                             line_color="red",
                             annotation_text=f"Labor Capacity: {labor_capacity}")
        
        fig.update_layout(
            title="Resource Utilization Over Time",
            xaxis_title="Days",
            yaxis_title="Workers Required",
            height=400,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def create_resource_allocation_pie(self, allocation_data: Dict) -> go.Figure:
        """Create pie chart showing resource allocation by resource type."""
        
        resource_type_totals = {
            'Labor': 0,
            'Equipment': 0,
            'Materials': 0,
            'Other': 0
        }
        
        for task_id, task_info in allocation_data.items():
            if 'resource_assignments' in task_info:
                for resource_id, amount in task_info['resource_assignments'].items():
                    resource_lower = resource_id.lower()
                    if 'worker' in resource_lower or 'labor' in resource_lower:
                        resource_type_totals['Labor'] += amount
                    elif 'equipment' in resource_lower or 'crane' in resource_lower or 'excavator' in resource_lower:
                        resource_type_totals['Equipment'] += amount
                    elif 'material' in resource_lower or 'cement' in resource_lower or 'steel' in resource_lower or 'wood' in resource_lower:
                        resource_type_totals['Materials'] += amount
                    else:
                        resource_type_totals['Other'] += amount
            elif 'labor_required' in task_info:
                resource_type_totals['Labor'] += task_info['labor_required']
                resource_type_totals['Equipment'] += task_info['labor_required'] * 0.2
                resource_type_totals['Materials'] += task_info['labor_required'] * 0.5
        
        filtered_types = {k: v for k, v in resource_type_totals.items() if v > 0}
        
        if not filtered_types:
            fig = go.Figure(data=[go.Pie(labels=['No Data'], values=[1])])
            fig.update_layout(title="No Resource Allocation Data Available")
            return fig
        
        colors = {
            'Labor': '#3498db',      # Blue
            'Equipment': '#e67e22',  # Orange
            'Materials': '#2ecc71',  # Green
            'Other': '#95a5a6'       # Gray
        }
        
        pie_colors = [colors.get(label, '#95a5a6') for label in filtered_types.keys()]
        
        fig = go.Figure(data=[go.Pie(
            labels=list(filtered_types.keys()),
            values=list(filtered_types.values()),
            hole=0.4,
            marker=dict(colors=pie_colors),
            textinfo='label+percent+value',
            textposition='auto',
            hovertemplate='<b>%{label}</b><br>Units: %{value:.1f}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title="Resource Allocation by Type",
            height=500,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.05
            )
        )
        
        return fig
    
    def create_algorithm_comparison_chart(self, results_dict: Dict) -> go.Figure:
        """Create a comprehensive comparison chart between optimization algorithms."""
        
        algorithms = []
        makespans = []
        costs = []
        solve_times = []
        
        for algorithm, result in results_dict.items():
            algorithms.append(algorithm)
            makespans.append(result.get('makespan', 0))
            costs.append(result.get('total_cost', 0))
            
            if 'solver_stats' in result:
                solve_times.append(result['solver_stats'].get('solve_time', 0))
            elif 'generations' in result:
                solve_times.append(result.get('generations', 0) * 0.01)
            else:
                solve_times.append(0)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Project Makespan (days)',
                'Total Cost ($)',
                'Solve Time (seconds)',
                'Efficiency Score'
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        fig.add_trace(
            go.Bar(
                x=algorithms,
                y=makespans,
                name='Makespan',
                marker_color='#3498db',
                text=[f"{m:.1f}d" for m in makespans],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=algorithms,
                y=costs,
                name='Cost',
                marker_color='#e74c3c',
                text=[f"${c:,.0f}" for c in costs],
                textposition='outside'
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=algorithms,
                y=solve_times,
                name='Solve Time',
                marker_color='#f39c12',
                text=[f"{t:.2f}s" for t in solve_times],
                textposition='outside'
            ),
            row=2, col=1
        )
        
        if makespans and costs:
            max_makespan = max(makespans) if max(makespans) > 0 else 1
            max_cost = max(costs) if max(costs) > 0 else 1
            efficiency_scores = [
                ((m / max_makespan) + (c / max_cost)) / 2 * 100
                for m, c in zip(makespans, costs)
            ]
            
            fig.add_trace(
                go.Bar(
                    x=algorithms,
                    y=efficiency_scores,
                    name='Efficiency',
                    marker_color='#2ecc71',
                    text=[f"{e:.1f}%" for e in efficiency_scores],
                    textposition='outside'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            title_text="Algorithm Performance Comparison",
            title_x=0.5
        )
        
        fig.update_xaxes(title_text="Algorithm", row=1, col=1)
        fig.update_xaxes(title_text="Algorithm", row=1, col=2)
        fig.update_xaxes(title_text="Algorithm", row=2, col=1)
        fig.update_xaxes(title_text="Algorithm", row=2, col=2)
        
        fig.update_yaxes(title_text="Days", row=1, col=1)
        fig.update_yaxes(title_text="Dollars", row=1, col=2)
        fig.update_yaxes(title_text="Seconds", row=2, col=1)
        fig.update_yaxes(title_text="Score", row=2, col=2)
        
        return fig
    
    def create_resource_breakdown_comparison(self, results_dict: Dict) -> go.Figure:
        """Compare resource allocation breakdown between algorithms."""
        
        algorithm_data = {}
        
        for algorithm, result in results_dict.items():
            resource_totals = {'Labor': 0, 'Equipment': 0, 'Materials': 0}
            
            schedule_data = result.get('schedule', result.get('allocation', {}))
            
            for task_id, task_info in schedule_data.items():
                if 'resource_assignments' in task_info:
                    for resource_id, amount in task_info['resource_assignments'].items():
                        resource_lower = resource_id.lower()
                        if 'worker' in resource_lower or 'labor' in resource_lower:
                            resource_totals['Labor'] += amount
                        elif 'equipment' in resource_lower or 'crane' in resource_lower or 'excavator' in resource_lower:
                            resource_totals['Equipment'] += amount
                        elif 'material' in resource_lower or 'cement' in resource_lower or 'steel' in resource_lower:
                            resource_totals['Materials'] += amount
                elif 'labor_required' in task_info:
                    resource_totals['Labor'] += task_info['labor_required']
                    resource_totals['Equipment'] += task_info['labor_required'] * 0.2
                    resource_totals['Materials'] += task_info['labor_required'] * 0.5
            
            algorithm_data[algorithm] = resource_totals
        
        fig = go.Figure()
        
        resource_types = ['Labor', 'Equipment', 'Materials']
        colors = {'Labor': '#3498db', 'Equipment': '#e67e22', 'Materials': '#2ecc71'}
        
        for resource_type in resource_types:
            values = [algorithm_data[alg][resource_type] for alg in algorithm_data.keys()]
            fig.add_trace(go.Bar(
                name=resource_type,
                x=list(algorithm_data.keys()),
                y=values,
                marker_color=colors[resource_type],
                text=[f"{v:.1f}" for v in values],
                textposition='outside'
            ))
        
        fig.update_layout(
            title='Resource Allocation Comparison by Type',
            xaxis_title='Algorithm',
            yaxis_title='Resource Units',
            barmode='group',
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_timeline_comparison(self, schedules: Dict[str, Dict]) -> go.Figure:
        """Compare multiple schedules side by side."""
        
        fig = make_subplots(
            rows=len(schedules),
            cols=1,
            subplot_titles=list(schedules.keys()),
            vertical_spacing=0.05
        )
        
        colors = px.colors.qualitative.Set1
        
        for row, (schedule_name, schedule_data) in enumerate(schedules.items(), 1):
            for i, (task_id, task_info) in enumerate(schedule_data.items()):
                fig.add_trace(
                    go.Scatter(
                        x=[task_info['start_time'], task_info['end_time']],
                        y=[task_info.get('task_name', task_id)] * 2,
                        mode='lines+markers',
                        name=f"{schedule_name} - {task_id}",
                        line=dict(width=8, color=colors[i % len(colors)]),
                        showlegend=row == 1  # Only show legend for first row
                    ),
                    row=row, col=1
                )
                
        fig.update_layout(
            title="Schedule Comparison",
            height=200 * len(schedules),
            hovermode='closest'
        )
        
        return fig
    
    def create_performance_dashboard(self, results: Dict) -> go.Figure:
        """Create a comprehensive performance dashboard."""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Project Timeline", "Resource Efficiency", 
                          "Cost & Schedule Metrics", "Algorithm Convergence"),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        if 'schedule' in results or 'allocation' in results:
            schedule_data = results.get('schedule', results.get('allocation', {}))
            
            task_ids = []
            start_times = []
            durations = []
            
            for task_id, task_info in schedule_data.items():
                task_ids.append(task_info.get('task_name', task_id))
                start_times.append(task_info['start_time'])
                durations.append(task_info['end_time'] - task_info['start_time'])
            
            fig.add_trace(
                go.Bar(
                    y=task_ids,
                    x=durations,
                    base=start_times,
                    orientation='h',
                    name='Task Duration',
                    marker=dict(color='#3498db'),
                    showlegend=False,
                    hovertemplate='<b>%{y}</b><br>Start: Day %{base}<br>Duration: %{x} days<extra></extra>'
                ),
                row=1, col=1
            )
                
        resource_totals = {}
        
        if 'schedule' in results or 'allocation' in results:
            schedule_data = results.get('schedule', results.get('allocation', {}))
            
            for task_id, task_info in schedule_data.items():
                if 'resource_assignments' in task_info:
                    for resource_id, amount in task_info['resource_assignments'].items():
                        if resource_id not in resource_totals:
                            resource_totals[resource_id] = 0
                        duration = task_info['end_time'] - task_info['start_time']
                        resource_totals[resource_id] += amount * duration
                elif 'labor_required' in task_info:
                    if 'Labor' not in resource_totals:
                        resource_totals['Labor'] = 0
                    duration = task_info['end_time'] - task_info['start_time']
                    resource_totals['Labor'] += task_info['labor_required'] * duration
        
        if not resource_totals and 'resource_usage' in results:
            resource_totals = results['resource_usage']
        
        if resource_totals:
            resource_totals = {k: v for k, v in resource_totals.items() if v > 0}
            sorted_resources = sorted(resource_totals.items(), key=lambda x: x[1], reverse=True)
            
            resource_names = [name for name, _ in sorted_resources]
            resource_usage = [value for _, value in sorted_resources]
            
            fig.add_trace(
                go.Bar(
                    x=resource_names, 
                    y=resource_usage, 
                    name="Usage",
                    marker=dict(color='#e67e22'),
                    showlegend=False,
                    text=[f"{u:.0f}" for u in resource_usage],
                    textposition='outside'
                ),
                row=1, col=2
            )
            
        metrics = []
        values = []
        
        if 'makespan' in results:
            metrics.append('Makespan<br>(days)')
            values.append(results['makespan'])
        
        if 'total_cost' in results:
            metrics.append('Cost<br>($1000s)')
            values.append(results['total_cost'] / 1000)
        
        if 'solver_stats' in results and 'solve_time' in results['solver_stats']:
            metrics.append('Solve Time<br>(seconds)')
            values.append(results['solver_stats']['solve_time'])
        elif 'generations' in results:
            metrics.append('Generations')
            values.append(results.get('generations', 0))
        
        if metrics:
            colors_map = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']
            fig.add_trace(
                go.Bar(
                    x=metrics, 
                    y=values,
                    marker=dict(color=colors_map[:len(metrics)]),
                    showlegend=False,
                    text=[f"{v:.1f}" for v in values],
                    textposition='outside'
                ),
                row=2, col=1
            )
            
        if 'fitness_history' in results and len(results['fitness_history']) > 0:
            generations = list(range(len(results['fitness_history'])))
            best_fitness = [h['best_fitness'] for h in results['fitness_history']]
            avg_fitness = [h['avg_fitness'] for h in results['fitness_history']]
            
            fig.add_trace(
                go.Scatter(
                    x=generations, 
                    y=best_fitness, 
                    mode='lines',
                    name="Best",
                    line=dict(color='#2ecc71', width=2),
                    showlegend=True
                ),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=generations, 
                    y=avg_fitness, 
                    mode='lines',
                    name="Average",
                    line=dict(color='#95a5a6', width=1, dash='dash'),
                    showlegend=True
                ),
                row=2, col=2
            )
        elif 'makespan' in results and 'total_cost' in results:
            status_text = results.get('status', 'UNKNOWN')
            
            kpi_labels = ['Status OK']
            kpi_values = [1 if status_text in ['OPTIMAL', 'FEASIBLE'] else 0]
            
            fig.add_trace(
                go.Bar(
                    x=kpi_labels,
                    y=kpi_values,
                    marker=dict(
                        color=['#2ecc71' if status_text in ['OPTIMAL', 'FEASIBLE'] else '#e74c3c']
                    ),
                    showlegend=False,
                    text=[status_text],
                    textposition='inside',
                    textfont=dict(size=14, color='white')
                ),
                row=2, col=2
            )
            
        fig.update_layout(
            title="Project Performance Dashboard",
            height=800,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5
            )
        )
        
        fig.update_xaxes(title_text="Project Day", row=1, col=1)
        fig.update_yaxes(title_text="Tasks", row=1, col=1)
        
        fig.update_xaxes(title_text="Resource Type", row=1, col=2)
        fig.update_yaxes(title_text="Units Used", row=1, col=2)
        
        fig.update_xaxes(title_text="Metric", row=2, col=1)
        fig.update_yaxes(title_text="Value", row=2, col=1)
        
        if 'fitness_history' in results and len(results['fitness_history']) > 0:
            fig.update_xaxes(title_text="Generation", row=2, col=2)
            fig.update_yaxes(title_text="Fitness Score", row=2, col=2)
        
        return fig
    
    def create_critical_path_visualization(self, schedule_data: Dict, dependencies: List[Tuple[str, str]]) -> go.Figure:
        """Visualize the critical path in the project schedule."""
        
        task_end_times = {task_id: task_info['end_time'] 
                         for task_id, task_info in schedule_data.items()}
        
        critical_end_task = max(task_end_times, key=task_end_times.get)
        
        critical_path = [critical_end_task]
        current_task = critical_end_task
        
        while True:
            predecessors = [dep[0] for dep in dependencies if dep[1] == current_task]
            if not predecessors:
                break
                
            latest_predecessor = max(predecessors, 
                                   key=lambda x: schedule_data[x]['end_time'])
            critical_path.insert(0, latest_predecessor)
            current_task = latest_predecessor
            
        fig = go.Figure()
        
        for task_id, task_info in schedule_data.items():
            color = 'red' if task_id in critical_path else 'lightblue'
            size = 20 if task_id in critical_path else 15
            
            fig.add_trace(go.Scatter(
                x=[task_info['start_time']],
                y=[task_id],
                mode='markers+text',
                text=[task_info.get('task_name', task_id)],
                textposition="middle right",
                marker=dict(size=size, color=color),
                name='Critical Path' if task_id in critical_path else 'Regular Task',
                showlegend=task_id == critical_path[0] or (task_id not in critical_path and task_id == list(schedule_data.keys())[0])
            ))
            
        for dep in dependencies:
            if dep[0] in schedule_data and dep[1] in schedule_data:
                x0, y0 = schedule_data[dep[0]]['end_time'], dep[0]
                x1, y1 = schedule_data[dep[1]]['start_time'], dep[1]
                
                line_color = 'red' if dep[0] in critical_path and dep[1] in critical_path else 'gray'
                line_width = 3 if dep[0] in critical_path and dep[1] in critical_path else 1
                
                fig.add_annotation(
                    x=x1, y=y1,
                    ax=x0, ay=y0,
                    xref='x', yref='y',
                    axref='x', ayref='y',
                    arrowhead=2,
                    arrowsize=1,
                    arrowcolor=line_color,
                    arrowwidth=line_width
                )
                
        fig.update_layout(
            title="Critical Path Analysis",
            xaxis_title="Days",
            yaxis_title="Tasks",
            height=max(400, len(schedule_data) * 50),
            showlegend=True,
            hovermode='closest'
        )
        
        return fig


def create_sample_visualization():
    """Create sample visualizations for testing."""
    
    schedule_data = {
        'T1': {'task_name': 'Site Preparation', 'start_time': 0, 'end_time': 5, 'duration': 5, 'labor_required': 3},
        'T2': {'task_name': 'Foundation', 'start_time': 5, 'end_time': 13, 'duration': 8, 'labor_required': 5},
        'T3': {'task_name': 'Framing', 'start_time': 13, 'end_time': 23, 'duration': 10, 'labor_required': 4},
        'T4': {'task_name': 'Roofing', 'start_time': 23, 'end_time': 29, 'duration': 6, 'labor_required': 3},
        'T5': {'task_name': 'Interior', 'start_time': 29, 'end_time': 41, 'duration': 12, 'labor_required': 6}
    }
    
    resources_data = {
        'workers': {'resource_type': 'labor', 'capacity': 10}
    }
    
    visualizer = ScheduleVisualizer()
    
    gantt_fig = visualizer.create_gantt_chart(schedule_data)
    resource_fig = visualizer.create_resource_utilization_chart(schedule_data, resources_data)
    pie_fig = visualizer.create_resource_allocation_pie(schedule_data)
    
    return gantt_fig, resource_fig, pie_fig


if __name__ == "__main__":
    gantt_fig, resource_fig, pie_fig = create_sample_visualization()
    
    print("Sample visualizations created successfully!")
    print("Use gantt_fig.show(), resource_fig.show(), pie_fig.show() to display them.")