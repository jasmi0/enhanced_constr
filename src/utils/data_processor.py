"""
Data Processing Utilities for Construction Scheduling System.

This module handles data ingestion, validation, feature engineering,
and preprocessing for construction project data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import json
import re
from pathlib import Path


class DataValidator:
    """Validates construction project data for consistency and completeness."""
    
    @staticmethod
    def validate_tasks_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate tasks DataFrame structure and content."""
        errors = []
        
        required_cols = ['task_id', 'task_name', 'duration', 'labor_required']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            
        if df.empty:
            errors.append("Tasks data is empty")
            return False, errors
            
        if 'duration' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['duration']):
                errors.append("Duration column must be numeric")
            elif (df['duration'] <= 0).any():
                errors.append("Duration must be positive")
                
        if 'labor_required' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['labor_required']):
                errors.append("Labor required column must be numeric")
            elif (df['labor_required'] < 0).any():
                errors.append("Labor required must be non-negative")
                
        if df['task_id'].duplicated().any():
            errors.append("Duplicate task IDs found")
            
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_resources_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate resources DataFrame structure and content."""
        errors = []
        
        required_cols = ['resource_id', 'resource_type', 'capacity']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            
        if df.empty:
            errors.append("Resources data is empty")
            return False, errors
            
        if 'capacity' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['capacity']):
                errors.append("Capacity column must be numeric")
            elif (df['capacity'] <= 0).any():
                errors.append("Capacity must be positive")
                
        if df['resource_id'].duplicated().any():
            errors.append("Duplicate resource IDs found")
            
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_dependencies(tasks_df: pd.DataFrame, dependencies: List[Tuple[str, str]]) -> Tuple[bool, List[str]]:
        """Validate task dependencies for circular references and invalid task IDs."""
        errors = []
        task_ids = set(tasks_df['task_id'].tolist())
        
        for dep in dependencies:
            if dep[0] not in task_ids:
                errors.append(f"Dependency references non-existent task: {dep[0]}")
            if dep[1] not in task_ids:
                errors.append(f"Dependency references non-existent task: {dep[1]}")
                
        def has_cycle(graph: Dict[str, List[str]]) -> bool:
            visited = set()
            rec_stack = set()
            
            def dfs(node: str) -> bool:
                if node in rec_stack:
                    return True
                if node in visited:
                    return False
                    
                visited.add(node)
                rec_stack.add(node)
                
                for neighbor in graph.get(node, []):
                    if dfs(neighbor):
                        return True
                        
                rec_stack.remove(node)
                return False
            
            for node in graph:
                if node not in visited:
                    if dfs(node):
                        return True
            return False
        
        dep_graph = {task_id: [] for task_id in task_ids}
        for predecessor, successor in dependencies:
            if predecessor in dep_graph:
                dep_graph[predecessor].append(successor)
                
        if has_cycle(dep_graph):
            errors.append("Circular dependency detected")
            
        return len(errors) == 0, errors


class DataProcessor:
    """Main data processing class for construction scheduling data."""
    
    def __init__(self):
        self.validator = DataValidator()
        self.tasks_df = None
        self.resources_df = None
        self.weather_df = None
        self.dependencies = []
        
    def load_from_csv(self, tasks_file: str, resources_file: str = None, 
                     weather_file: str = None, dependencies_file: str = None) -> bool:
        """Load data from CSV files."""
        try:
            self.tasks_df = pd.read_csv(tasks_file)
            
            is_valid, errors = self.validator.validate_tasks_data(self.tasks_df)
            if not is_valid:
                raise ValueError(f"Invalid tasks data: {errors}")
                
            if resources_file:
                self.resources_df = pd.read_csv(resources_file)
                is_valid, errors = self.validator.validate_resources_data(self.resources_df)
                if not is_valid:
                    raise ValueError(f"Invalid resources data: {errors}")
                    
            if weather_file:
                self.weather_df = pd.read_csv(weather_file)
                
            if dependencies_file:
                deps_df = pd.read_csv(dependencies_file)
                self.dependencies = list(zip(deps_df['predecessor'], deps_df['successor']))
                
                is_valid, errors = self.validator.validate_dependencies(self.tasks_df, self.dependencies)
                if not is_valid:
                    raise ValueError(f"Invalid dependencies: {errors}")
                    
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def load_from_dataframes(self, tasks_df: pd.DataFrame, resources_df: pd.DataFrame = None,
                           weather_df: pd.DataFrame = None, dependencies: List[Tuple[str, str]] = None) -> bool:
        """Load data from pandas DataFrames."""
        try:
            is_valid, errors = self.validator.validate_tasks_data(tasks_df)
            if not is_valid:
                raise ValueError(f"Invalid tasks data: {errors}")
            self.tasks_df = tasks_df.copy()
            
            if resources_df is not None:
                is_valid, errors = self.validator.validate_resources_data(resources_df)
                if not is_valid:
                    raise ValueError(f"Invalid resources data: {errors}")
                self.resources_df = resources_df.copy()
                
            if weather_df is not None:
                self.weather_df = weather_df.copy()
                
            if dependencies:
                is_valid, errors = self.validator.validate_dependencies(tasks_df, dependencies)
                if not is_valid:
                    raise ValueError(f"Invalid dependencies: {errors}")
                self.dependencies = dependencies
                
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def extract_features(self) -> Dict:
        """Extract and engineer features from the loaded data."""
        if self.tasks_df is None:
            raise ValueError("No tasks data loaded")
            
        features = {}
        
        features['total_tasks'] = len(self.tasks_df)
        features['total_duration'] = self.tasks_df['duration'].sum()
        features['avg_duration'] = self.tasks_df['duration'].mean()
        features['max_duration'] = self.tasks_df['duration'].max()
        features['min_duration'] = self.tasks_df['duration'].min()
        
        features['total_labor'] = self.tasks_df['labor_required'].sum()
        features['avg_labor'] = self.tasks_df['labor_required'].mean()
        features['peak_labor'] = self.tasks_df['labor_required'].max()
        
        self.tasks_df['complexity'] = self.tasks_df['duration'] * self.tasks_df['labor_required']
        features['total_complexity'] = self.tasks_df['complexity'].sum()
        features['avg_complexity'] = self.tasks_df['complexity'].mean()
        
        if self.dependencies:
            features['dependency_count'] = len(self.dependencies)
            features['dependency_density'] = len(self.dependencies) / max(len(self.tasks_df) ** 2, 1)
        else:
            features['dependency_count'] = 0
            features['dependency_density'] = 0
            
        if self.resources_df is not None:
            features['total_resources'] = len(self.resources_df)
            if 'capacity' in self.resources_df.columns:
                features['total_resource_capacity'] = self.resources_df['capacity'].sum()
                features['avg_resource_capacity'] = self.resources_df['capacity'].mean()
                
        return features
    
    def preprocess_for_scheduler(self) -> Dict:
        """Prepare data for the OR-Tools scheduler."""
        from src.models.scheduler import Task, Resource, WeatherConstraint
        
        if self.tasks_df is None:
            raise ValueError("No tasks data loaded")
            
        tasks = {}
        for _, row in self.tasks_df.iterrows():
            materials_needed = {}
            if 'materials_needed' in row and pd.notna(row['materials_needed']):
                try:
                    materials_needed = json.loads(row['materials_needed'])
                except:
                    if isinstance(row['materials_needed'], str):
                        for item in row['materials_needed'].split(','):
                            if ':' in item:
                                material, amount = item.strip().split(':')
                                materials_needed[material.strip()] = int(amount.strip())
            
            task_dependencies = []
            for dep in self.dependencies:
                if dep[1] == row['task_id']:  # This task depends on dep[0]
                    task_dependencies.append(dep[0])
                    
            earliest_start = int(row.get('earliest_start', 0))
            latest_finish = None
            if 'latest_finish' in row and pd.notna(row['latest_finish']):
                latest_finish = int(row['latest_finish'])
                
            task = Task(
                task_id=row['task_id'],
                name=row['task_name'],
                duration=int(row['duration']),
                labor_required=int(row['labor_required']),
                materials_needed=materials_needed,
                dependencies=task_dependencies,
                earliest_start=earliest_start,
                latest_finish=latest_finish
            )
            tasks[row['task_id']] = task
            
        resources = {}
        if self.resources_df is not None:
            for _, row in self.resources_df.iterrows():
                availability_periods = None
                if 'availability_periods' in row and pd.notna(row['availability_periods']):
                    try:
                        availability_periods = json.loads(row['availability_periods'])
                    except:
                        pass
                
                cost_per_unit = None
                if 'cost_per_unit' in row and pd.notna(row['cost_per_unit']):
                    cost_per_unit = float(row['cost_per_unit'])
                        
                resource = Resource(
                    resource_id=row['resource_id'],
                    resource_type=row['resource_type'],
                    capacity=int(row['capacity']),
                    availability_periods=availability_periods,
                    cost_per_unit=cost_per_unit
                )
                resources[row['resource_id']] = resource
                
        weather_constraints = []
        if self.weather_df is not None:
            for _, row in self.weather_df.iterrows():
                affected_tasks = []
                if 'affected_tasks' in row and pd.notna(row['affected_tasks']):
                    affected_tasks = row['affected_tasks'].split(',')
                    affected_tasks = [task.strip() for task in affected_tasks]
                    
                blocked_periods = []
                if 'blocked_periods' in row and pd.notna(row['blocked_periods']):
                    try:
                        blocked_periods = json.loads(row['blocked_periods'])
                    except:
                        pass
                        
                weather_constraint = WeatherConstraint(
                    weather_type=row.get('weather_type', 'unknown'),
                    affected_tasks=affected_tasks,
                    blocked_periods=blocked_periods
                )
                weather_constraints.append(weather_constraint)
                
        return {
            'tasks': tasks,
            'resources': resources,
            'weather_constraints': weather_constraints
        }
    
    def preprocess_for_genetic_algorithm(self) -> Dict:
        """Prepare data for the genetic algorithm optimizer."""
        if self.tasks_df is None:
            raise ValueError("No tasks data loaded")
            
        task_data = {}
        for _, row in self.tasks_df.iterrows():
            resource_requirements = {}
            
            for col in self.tasks_df.columns:
                if col.endswith('_min') or col.endswith('_max'):
                    resource_type = col.rsplit('_', 1)[0]
                    if col.endswith('_min'):
                        if resource_type + '_max' in self.tasks_df.columns:
                            min_val = int(row[col]) if pd.notna(row[col]) else 0
                            max_val = int(row[resource_type + '_max']) if pd.notna(row[resource_type + '_max']) else min_val
                            resource_requirements[resource_type] = (min_val, max_val)
                            
            if not resource_requirements and 'labor_required' in row:
                labor_req = int(row['labor_required'])
                resource_requirements['workers'] = (labor_req, labor_req)
                
            priority = float(row.get('priority', 1.0))
            deadline = None
            if 'deadline' in row and pd.notna(row['deadline']):
                deadline = int(row['deadline'])
                
            task_data[row['task_id']] = {
                'resource_requirements': resource_requirements,
                'duration': int(row['duration']),
                'priority': priority,
                'deadline': deadline
            }
            
        resource_data = {}
        if self.resources_df is not None:
            for _, row in self.resources_df.iterrows():
                cost_per_unit = float(row.get('cost_per_unit', 1.0))
                resource_data[row['resource_id']] = {
                    'total_capacity': int(row['capacity']),
                    'cost_per_unit': cost_per_unit
                }
                
        return {
            'tasks': task_data,
            'resources': resource_data
        }
    
    def export_results(self, results: Dict, output_file: str) -> bool:
        """Export optimization results to various formats."""
        try:
            if output_file.endswith('.csv'):
                if 'schedule' in results:
                    schedule_data = []
                    for task_id, task_info in results['schedule'].items():
                        schedule_data.append({
                            'task_id': task_id,
                            'task_name': task_info.get('task_name', ''),
                            'start_time': task_info['start_time'],
                            'end_time': task_info['end_time'],
                            'duration': task_info.get('duration', ''),
                            'labor_required': task_info.get('labor_required', '')
                        })
                    df = pd.DataFrame(schedule_data)
                    df.to_csv(output_file, index=False)
                    
            elif output_file.endswith('.json'):
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                    
            return True
            
        except Exception as e:
            print(f"Error exporting results: {e}")
            return False


def create_sample_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create sample data for testing."""
    tasks_data = {
        'task_id': ['T1', 'T2', 'T3', 'T4', 'T5'],
        'task_name': ['Site Preparation', 'Foundation', 'Framing', 'Roofing', 'Interior'],
        'duration': [5, 8, 10, 6, 12],
        'labor_required': [3, 5, 4, 3, 6],
        'materials_needed': [
            '{"cement": 10}',
            '{"cement": 50, "steel": 20}',
            '{"wood": 100, "steel": 30}',
            '{"roofing_materials": 40}',
            '{"finishing_materials": 80}'
        ],
        'earliest_start': [0, 0, 0, 0, 0],
        'latest_finish': [10, 20, 35, 45, 60],
        'priority': [1.0, 1.0, 0.8, 0.9, 0.6]
    }
    
    resources_data = {
        'resource_id': ['workers', 'cement', 'steel', 'wood', 'roofing_materials', 'finishing_materials'],
        'resource_type': ['labor', 'material', 'material', 'material', 'material', 'material'],
        'capacity': [10, 100, 100, 200, 50, 100],
        'cost_per_unit': [100, 1, 2, 1.5, 3, 2]
    }
    
    weather_data = {
        'weather_type': ['rain', 'snow'],
        'affected_tasks': ['T4', 'T3,T4'],
        'blocked_periods': ['[[15, 20], [35, 40]]', '[[50, 55]]']
    }
    
    return (pd.DataFrame(tasks_data), 
            pd.DataFrame(resources_data), 
            pd.DataFrame(weather_data))


if __name__ == "__main__":
    processor = DataProcessor()
    
    tasks_df, resources_df, weather_df = create_sample_data()
    dependencies = [('T1', 'T2'), ('T2', 'T3'), ('T3', 'T4'), ('T3', 'T5')]
    
    success = processor.load_from_dataframes(tasks_df, resources_df, weather_df, dependencies)
    
    if success:
        print("Data loaded successfully!")
        
        features = processor.extract_features()
        print(f"\nFeatures extracted: {len(features)} features")
        for key, value in features.items():
            print(f"  {key}: {value}")
            
        scheduler_data = processor.preprocess_for_scheduler()
        print(f"\nScheduler data prepared: {len(scheduler_data['tasks'])} tasks, "
              f"{len(scheduler_data['resources'])} resources")
              
        ga_data = processor.preprocess_for_genetic_algorithm()
        print(f"GA data prepared: {len(ga_data['tasks'])} tasks, "
              f"{len(ga_data['resources'])} resources")
    else:
        print("Failed to load data")