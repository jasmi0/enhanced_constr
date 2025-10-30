"""
Construction Scheduling Optimizer using OR-Tools CP-SAT solver.

This module handles constraint-based scheduling for construction projects,
considering factors like labor availability, materials, weather, and task dependencies.
"""

from ortools.sat.python import cp_model
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np


class Task:
    """Represents a construction task with all its properties."""
    
    def __init__(self, task_id: str, name: str, duration: int, labor_required: int,
                 materials_needed: Dict[str, int], dependencies: List[str] = None,
                 earliest_start: int = 0, latest_finish: int = None):
        self.task_id = task_id
        self.name = name
        self.duration = duration
        self.labor_required = labor_required
        self.materials_needed = materials_needed or {}
        self.dependencies = dependencies or []
        self.earliest_start = earliest_start
        self.latest_finish = latest_finish


class Resource:
    """Represents available resources (labor, materials, equipment)."""
    
    def __init__(self, resource_id: str, resource_type: str, capacity: int,
                 availability_periods: List[Tuple[int, int]] = None, cost_per_unit: float = None):
        self.resource_id = resource_id
        self.resource_type = resource_type
        self.capacity = capacity
        self.availability_periods = availability_periods or [(0, 1000)]  # Default: always available
        self.cost_per_unit = cost_per_unit


class WeatherConstraint:
    """Represents weather constraints affecting certain tasks."""
    
    def __init__(self, weather_type: str, affected_tasks: List[str],
                 blocked_periods: List[Tuple[int, int]]):
        self.weather_type = weather_type
        self.affected_tasks = affected_tasks
        self.blocked_periods = blocked_periods


class ConstructionScheduler:
    """Main class for construction project scheduling optimization."""
    
    def __init__(self, project_horizon: int = 365):
        self.project_horizon = project_horizon
        self.model = cp_model.CpModel()
        self.tasks = {}
        self.resources = {}
        self.weather_constraints = []
        self.task_intervals = {}
        self.task_starts = {}
        self.task_ends = {}
        
    def _get_material_cost(self, material: str) -> float:
        """Get the cost per unit for a material."""
        material_costs = {
            'cement': 1.0,
            'steel': 2.0,
            'wood': 1.5,
            'roofing_materials': 3.0,
            'electrical_materials': 2.5,
            'plumbing_materials': 2.0,
            'finishing_materials': 2.0
        }
        
        for resource_id, resource in self.resources.items():
            if resource_id == material and hasattr(resource, 'cost_per_unit'):
                return resource.cost_per_unit
                
        return material_costs.get(material, 1.0)
        
    def add_task(self, task: Task):
        """Add a task to the scheduling problem."""
        self.tasks[task.task_id] = task
        
    def add_resource(self, resource: Resource):
        """Add a resource to the scheduling problem."""
        self.resources[resource.resource_id] = resource
        
    def add_weather_constraint(self, weather_constraint: WeatherConstraint):
        """Add weather constraints that block certain tasks during specific periods."""
        self.weather_constraints.append(weather_constraint)
        
    def create_schedule_variables(self):
        """Create decision variables for task scheduling."""
        for task_id, task in self.tasks.items():
            start_var = self.model.NewIntVar(
                task.earliest_start,
                self.project_horizon - task.duration,
                f'start_{task_id}'
            )
            self.task_starts[task_id] = start_var
            
            end_var = self.model.NewIntVar(
                task.earliest_start + task.duration,
                self.project_horizon,
                f'end_{task_id}'
            )
            self.task_ends[task_id] = end_var
            
            interval_var = self.model.NewIntervalVar(
                start_var, task.duration, end_var, f'interval_{task_id}'
            )
            self.task_intervals[task_id] = interval_var
            
            if task.latest_finish is not None:
                self.model.Add(end_var <= task.latest_finish)
                
    def add_dependency_constraints(self):
        """Add precedence constraints based on task dependencies."""
        for task_id, task in self.tasks.items():
            for dependency_id in task.dependencies:
                if dependency_id in self.task_ends:
                    self.model.Add(
                        self.task_starts[task_id] >= self.task_ends[dependency_id]
                    )
                    
    def add_resource_constraints(self):
        """Add resource capacity constraints."""
        for resource_id, resource in self.resources.items():
            if resource.resource_type == 'labor':
                labor_intervals = []
                labor_demands = []
                
                for task_id, task in self.tasks.items():
                    if task.labor_required > 0:
                        labor_intervals.append(self.task_intervals[task_id])
                        labor_demands.append(task.labor_required)
                
                if labor_intervals:
                    self.model.AddCumulative(
                        labor_intervals, labor_demands, resource.capacity
                    )
            
            elif resource.resource_type == 'material':
                for task_id, task in self.tasks.items():
                    if resource_id in task.materials_needed:
                        required_amount = task.materials_needed[resource_id]
                        if required_amount > resource.capacity:
                            self.model.Add(self.task_starts[task_id] >= self.project_horizon)
                            
    def add_weather_constraints(self):
        """Add weather-related constraints that prevent certain tasks during bad weather."""
        for weather_constraint in self.weather_constraints:
            for task_id in weather_constraint.affected_tasks:
                if task_id in self.task_intervals:
                    task_interval = self.task_intervals[task_id]
                    task_start = self.task_starts[task_id]
                    task_end = self.task_ends[task_id]
                    
                    for blocked_start, blocked_end in weather_constraint.blocked_periods:
                        before_blocked = self.model.NewBoolVar(f'before_blocked_{task_id}_{blocked_start}')
                        after_blocked = self.model.NewBoolVar(f'after_blocked_{task_id}_{blocked_end}')
                        
                        self.model.Add(task_end <= blocked_start).OnlyEnforceIf(before_blocked)
                        self.model.Add(task_start >= blocked_end).OnlyEnforceIf(after_blocked)
                        self.model.Add(before_blocked + after_blocked >= 1)
                        
    def set_objective(self, objective_type: str = 'minimize_makespan'):
        """Set the optimization objective."""
        if not self.task_ends:
            raise ValueError("Task variables must be created before setting objective. Call create_schedule_variables() first.")
            
        if objective_type == 'minimize_makespan':
            makespan = self.model.NewIntVar(0, self.project_horizon, 'makespan')
            for task_id in self.tasks:
                self.model.Add(makespan >= self.task_ends[task_id])
            self.model.Minimize(makespan)
            
        elif objective_type == 'minimize_total_duration':
            total_duration = sum(self.task_ends[task_id] for task_id in self.tasks)
            self.model.Minimize(total_duration)
            
    def solve(self, time_limit_seconds: int = 300, objective_type: str = 'minimize_makespan') -> Dict:
        """Solve the scheduling problem and return results."""
        self.create_schedule_variables()
        self.add_dependency_constraints()
        self.add_resource_constraints()
        self.add_weather_constraints()
        self.set_objective(objective_type)
        
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit_seconds
        
        status = solver.Solve(self.model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            schedule = {}
            total_cost = 0.0
            resource_usage = {}
            
            for task_id in self.tasks:
                start_time = solver.Value(self.task_starts[task_id])
                end_time = solver.Value(self.task_ends[task_id])
                task = self.tasks[task_id]
                
                schedule[task_id] = {
                    'task_name': task.name,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': task.duration,
                    'labor_required': task.labor_required
                }
                
                labor_cost = task.labor_required * task.duration * 100
                total_cost += labor_cost
                
                for material, amount in task.materials_needed.items():
                    material_cost = amount * self._get_material_cost(material)
                    total_cost += material_cost
                    
                    if material in resource_usage:
                        resource_usage[material] += amount
                    else:
                        resource_usage[material] = amount
                
                if 'workers' in resource_usage:
                    resource_usage['workers'] = max(resource_usage['workers'], task.labor_required)
                else:
                    resource_usage['workers'] = task.labor_required
                
            return {
                'status': 'OPTIMAL' if status == cp_model.OPTIMAL else 'FEASIBLE',
                'schedule': schedule,
                'makespan': max(schedule[task_id]['end_time'] for task_id in schedule),
                'total_cost': total_cost,
                'resource_usage': resource_usage,
                'solver_stats': {
                    'solve_time': solver.WallTime(),
                    'num_conflicts': solver.NumConflicts(),
                    'num_branches': solver.NumBranches()
                }
            }
        else:
            return {
                'status': 'INFEASIBLE' if status == cp_model.INFEASIBLE else 'UNKNOWN',
                'schedule': {},
                'makespan': None,
                'total_cost': 0.0,
                'resource_usage': {},
                'solver_stats': {
                    'solve_time': solver.WallTime(),
                    'num_conflicts': solver.NumConflicts(),
                    'num_branches': solver.NumBranches()
                }
            }


def create_sample_problem() -> ConstructionScheduler:
    """Create a sample construction scheduling problem for testing."""
    scheduler = ConstructionScheduler(project_horizon=100)
    
    tasks = [
        Task('T1', 'Site Preparation', 5, 3, {'cement': 10}, []),
        Task('T2', 'Foundation', 8, 5, {'cement': 50, 'steel': 20}, ['T1']),
        Task('T3', 'Framing', 10, 4, {'wood': 100, 'steel': 30}, ['T2']),
        Task('T4', 'Roofing', 6, 3, {'roofing_materials': 40}, ['T3']),
        Task('T5', 'Electrical', 7, 2, {'electrical_materials': 25}, ['T3']),
        Task('T6', 'Plumbing', 5, 2, {'plumbing_materials': 30}, ['T3']),
        Task('T7', 'Interior Finishing', 12, 6, {'finishing_materials': 80}, ['T4', 'T5', 'T6'])
    ]
    
    for task in tasks:
        scheduler.add_task(task)
    
    resources = [
        Resource('labor_team', 'labor', 10, cost_per_unit=100.0),
        Resource('cement', 'material', 100, cost_per_unit=1.0),
        Resource('steel', 'material', 100, cost_per_unit=2.0),
        Resource('wood', 'material', 200, cost_per_unit=1.5),
        Resource('roofing_materials', 'material', 50, cost_per_unit=3.0),
        Resource('electrical_materials', 'material', 50, cost_per_unit=2.5),
        Resource('plumbing_materials', 'material', 50, cost_per_unit=2.0),
        Resource('finishing_materials', 'material', 100, cost_per_unit=2.0)
    ]
    
    for resource in resources:
        scheduler.add_resource(resource)
    
    weather = WeatherConstraint('rain', ['T4'], [(15, 20), (35, 40)])
    scheduler.add_weather_constraint(weather)
    
    return scheduler


if __name__ == "__main__":
    scheduler = create_sample_problem()
    result = scheduler.solve()
    
    print(f"Optimization Status: {result['status']}")
    print(f"Project Makespan: {result['makespan']} days")
    print("\nSchedule:")
    for task_id, task_info in result['schedule'].items():
        print(f"{task_id}: {task_info['task_name']} - "
              f"Days {task_info['start_time']} to {task_info['end_time']} "
              f"(Duration: {task_info['duration']}, Labor: {task_info['labor_required']})")