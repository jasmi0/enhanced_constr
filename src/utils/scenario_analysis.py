"""
Scenario Analysis Module for Construction Scheduling System.

This module provides what-if analysis capabilities for construction projects,
allowing users to test different scenarios and compare outcomes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from copy import deepcopy
import json

from ..models.scheduler import ConstructionScheduler, Task, Resource, WeatherConstraint
from ..models.genetic_optimizer import ResourceOptimizer
from ..utils.data_processor import DataProcessor


class Scenario:
    """Represents a single scenario configuration."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.modifications = []
        self.results = None
        
    def add_modification(self, modification_type: str, **params):
        """Add a modification to this scenario."""
        self.modifications.append({
            'type': modification_type,
            'params': params
        })
        
    def apply_to_data_processor(self, data_processor: DataProcessor) -> DataProcessor:
        """Apply scenario modifications to a data processor and return modified copy."""
        # Create a deep copy to avoid modifying original data
        modified_processor = DataProcessor()
        
        # Copy original data
        if data_processor.tasks_df is not None:
            modified_processor.tasks_df = data_processor.tasks_df.copy()
        if data_processor.resources_df is not None:
            modified_processor.resources_df = data_processor.resources_df.copy()
        if data_processor.weather_df is not None:
            modified_processor.weather_df = data_processor.weather_df.copy()
        modified_processor.dependencies = data_processor.dependencies.copy()
        
        # Apply modifications
        for mod in self.modifications:
            self._apply_modification(modified_processor, mod)
            
        return modified_processor
    
    def _apply_modification(self, processor: DataProcessor, modification: Dict):
        """Apply a single modification to the data processor."""
        mod_type = modification['type']
        params = modification['params']
        
        print(f"    Applying: {mod_type}")
        
        if mod_type == 'change_labor_availability':
            # Modify labor capacity
            multiplier = params.get('multiplier', 1.0)
            if processor.resources_df is not None:
                mask = processor.resources_df['resource_type'] == 'labor'
                original_capacity = processor.resources_df.loc[mask, 'capacity'].sum()
                processor.resources_df.loc[mask, 'capacity'] = (
                    processor.resources_df.loc[mask, 'capacity'] * multiplier
                ).astype(int)
                new_capacity = processor.resources_df.loc[mask, 'capacity'].sum()
                print(f"      Labor capacity: {original_capacity} → {new_capacity} (×{multiplier})")
                
        elif mod_type == 'change_material_cost':
            # Modify material costs
            material = params.get('material')
            multiplier = params.get('multiplier', 1.0)
            if processor.resources_df is not None and material:
                mask = processor.resources_df['resource_id'] == material
                if 'cost_per_unit' in processor.resources_df.columns:
                    original_cost = processor.resources_df.loc[mask, 'cost_per_unit'].values
                    processor.resources_df.loc[mask, 'cost_per_unit'] *= multiplier
                    new_cost = processor.resources_df.loc[mask, 'cost_per_unit'].values
                    if len(original_cost) > 0:
                        print(f"      {material} cost: ${original_cost[0]:.2f} → ${new_cost[0]:.2f} (×{multiplier})")
                    else:
                        print(f"      WARNING: Material '{material}' not found!")
                    
        elif mod_type == 'add_weather_delay':
            # Add weather constraints
            affected_tasks = params.get('affected_tasks', [])
            delay_periods = params.get('delay_periods', [])
            weather_type = params.get('weather_type', 'additional_weather')
            
            print(f"      Adding weather delays for tasks: {affected_tasks}")
            print(f"      Delay periods: {delay_periods}")
            
            if processor.weather_df is not None:
                new_weather = pd.DataFrame({
                    'weather_type': [weather_type],
                    'affected_tasks': [','.join(affected_tasks)],
                    'blocked_periods': [json.dumps(delay_periods)]
                })
                processor.weather_df = pd.concat([processor.weather_df, new_weather], ignore_index=True)
            else:
                processor.weather_df = pd.DataFrame({
                    'weather_type': [weather_type],
                    'affected_tasks': [','.join(affected_tasks)],
                    'blocked_periods': [json.dumps(delay_periods)]
                })
                
        elif mod_type == 'change_task_duration':
            # Modify task durations
            task_id = params.get('task_id')
            multiplier = params.get('multiplier', 1.0)
            if processor.tasks_df is not None and task_id:
                mask = processor.tasks_df['task_id'] == task_id
                original_duration = processor.tasks_df.loc[mask, 'duration'].values
                processor.tasks_df.loc[mask, 'duration'] = (
                    processor.tasks_df.loc[mask, 'duration'] * multiplier
                ).astype(int)
                new_duration = processor.tasks_df.loc[mask, 'duration'].values
                if len(original_duration) > 0:
                    print(f"      {task_id} duration: {original_duration[0]} → {new_duration[0]} days (×{multiplier})")
                else:
                    print(f"      WARNING: Task '{task_id}' not found!")
                
        elif mod_type == 'change_task_priority':
            # Modify task priority
            task_id = params.get('task_id')
            new_priority = params.get('priority', 1.0)
            if processor.tasks_df is not None and task_id:
                mask = processor.tasks_df['task_id'] == task_id
                if 'priority' in processor.tasks_df.columns:
                    original_priority = processor.tasks_df.loc[mask, 'priority'].values
                    processor.tasks_df.loc[mask, 'priority'] = new_priority
                    if len(original_priority) > 0:
                        print(f"      {task_id} priority: {original_priority[0]} → {new_priority}")
                    else:
                        print(f"      WARNING: Task '{task_id}' not found!")
                    
        elif mod_type == 'add_resource_constraint':
            # Add or modify resource constraints
            resource_id = params.get('resource_id')
            new_capacity = params.get('capacity')
            if processor.resources_df is not None and resource_id and new_capacity:
                mask = processor.resources_df['resource_id'] == resource_id
                if mask.any():
                    original_capacity = processor.resources_df.loc[mask, 'capacity'].values[0]
                    processor.resources_df.loc[mask, 'capacity'] = new_capacity
                    print(f"      {resource_id} capacity: {original_capacity} → {new_capacity}")
                else:
                    # Add new resource
                    new_resource = pd.DataFrame({
                        'resource_id': [resource_id],
                        'resource_type': [params.get('resource_type', 'material')],
                        'capacity': [new_capacity],
                        'cost_per_unit': [params.get('cost_per_unit', 1.0)]
                    })
                    processor.resources_df = pd.concat([processor.resources_df, new_resource], ignore_index=True)
                    print(f"      Added new resource: {resource_id} (capacity={new_capacity})")


class ScenarioAnalyzer:
    """Main class for conducting scenario analysis."""
    
    def __init__(self, base_data_processor: DataProcessor):
        self.base_processor = base_data_processor
        self.scenarios = {}
        self.baseline_results = None
        
    def add_scenario(self, scenario: Scenario):
        """Add a scenario to the analysis."""
        self.scenarios[scenario.name] = scenario
        
    def create_predefined_scenarios(self) -> List[Scenario]:
        """Create a set of predefined scenarios for common what-if analyses."""
        scenarios = []
        
        # Labor shortage scenario
        labor_shortage = Scenario(
            "Labor Shortage", 
            "Reduced labor availability by 30%"
        )
        labor_shortage.add_modification('change_labor_availability', multiplier=0.7)
        scenarios.append(labor_shortage)
        
        # Labor abundance scenario
        labor_abundance = Scenario(
            "Labor Abundance", 
            "Increased labor availability by 50%"
        )
        labor_abundance.add_modification('change_labor_availability', multiplier=1.5)
        scenarios.append(labor_abundance)
        
        # Material cost increase scenario
        material_cost_increase = Scenario(
            "Material Cost Spike", 
            "Steel costs increased by 100%"
        )
        material_cost_increase.add_modification('change_material_cost', material='steel', multiplier=2.0)
        scenarios.append(material_cost_increase)
        
        # Weather delay scenario
        weather_delays = Scenario(
            "Extended Weather Delays", 
            "Additional rain periods affecting outdoor tasks"
        )
        weather_delays.add_modification(
            'add_weather_delay', 
            affected_tasks=['T1', 'T4'], 
            delay_periods=[[25, 30], [45, 50]],
            weather_type='extended_rain'
        )
        scenarios.append(weather_delays)
        
        # Rush job scenario
        rush_job = Scenario(
            "Rush Project", 
            "All task durations reduced by 20% (crash schedule)"
        )
        if self.base_processor.tasks_df is not None:
            for task_id in self.base_processor.tasks_df['task_id']:
                rush_job.add_modification('change_task_duration', task_id=task_id, multiplier=0.8)
        scenarios.append(rush_job)
        
        # High priority scenario
        priority_change = Scenario(
            "Priority Rebalancing", 
            "Roofing task set to highest priority"
        )
        priority_change.add_modification('change_task_priority', task_id='T4', priority=2.0)
        scenarios.append(priority_change)
        
        return scenarios
    
    def run_baseline(self, algorithm: str = 'ortools', **kwargs) -> Dict:
        """Run baseline optimization without any scenario modifications."""
        print(f"\n{'='*60}")
        print(f"RUNNING BASELINE OPTIMIZATION")
        print(f"{'='*60}")
        
        if algorithm.lower() == 'ortools':
            result = self._run_ortools_optimization(self.base_processor, **kwargs)
        elif algorithm.lower() == 'genetic':
            result = self._run_genetic_optimization(self.base_processor, **kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        print(f"Baseline Results:")
        print(f"  Status: {result.get('status')}")
        print(f"  Makespan: {result.get('makespan')} days")
        print(f"  Total Cost: ${result.get('total_cost'):,.2f}")
        print(f"{'='*60}\n")
        
        return result
    
    def run_scenario_analysis(self, scenarios: List[str] = None, 
                            algorithm: str = 'ortools', **kwargs) -> Dict[str, Dict]:
        """Run analysis for specified scenarios."""
        if scenarios is None:
            scenarios = list(self.scenarios.keys())
            
        results = {}
        
        # Run baseline if not already done
        if self.baseline_results is None:
            self.baseline_results = self.run_baseline(algorithm, **kwargs)
            
        results['Baseline'] = self.baseline_results
        
        # Run each scenario
        for scenario_name in scenarios:
            if scenario_name in self.scenarios:
                print(f"\n{'='*60}")
                print(f"RUNNING SCENARIO: {scenario_name}")
                print(f"{'='*60}")
                
                scenario = self.scenarios[scenario_name]
                print(f"Description: {scenario.description}")
                print(f"Modifications: {len(scenario.modifications)}")
                for i, mod in enumerate(scenario.modifications, 1):
                    print(f"  {i}. {mod['type']}: {mod['params']}")
                
                modified_processor = scenario.apply_to_data_processor(self.base_processor)
                
                # Debug: Check if modifications were applied
                print(f"\nData after modifications:")
                if modified_processor.tasks_df is not None:
                    print(f"  Tasks: {len(modified_processor.tasks_df)} tasks")
                    if 'duration' in modified_processor.tasks_df.columns:
                        print(f"  Average duration: {modified_processor.tasks_df['duration'].mean():.2f} days")
                if modified_processor.resources_df is not None:
                    print(f"  Resources: {len(modified_processor.resources_df)} resources")
                    labor_resources = modified_processor.resources_df[
                        modified_processor.resources_df['resource_type'] == 'labor'
                    ]
                    if not labor_resources.empty:
                        print(f"  Labor capacity: {labor_resources['capacity'].sum()}")
                
                try:
                    if algorithm.lower() == 'ortools':
                        scenario_result = self._run_ortools_optimization(modified_processor, **kwargs)
                    elif algorithm.lower() == 'genetic':
                        scenario_result = self._run_genetic_optimization(modified_processor, **kwargs)
                    else:
                        scenario_result = {'status': 'ERROR', 'error': f'Unknown algorithm: {algorithm}'}
                    
                    print(f"\nScenario Results:")
                    print(f"  Status: {scenario_result.get('status')}")
                    print(f"  Makespan: {scenario_result.get('makespan')} days")
                    print(f"  Total Cost: ${scenario_result.get('total_cost'):,.2f}")
                    
                    # Calculate changes from baseline
                    baseline_makespan = self.baseline_results.get('makespan', 0)
                    baseline_cost = self.baseline_results.get('total_cost', 0)
                    makespan_change = scenario_result.get('makespan', 0) - baseline_makespan
                    cost_change = scenario_result.get('total_cost', 0) - baseline_cost
                    
                    print(f"\nChanges from Baseline:")
                    print(f"  Makespan: {makespan_change:+.1f} days ({makespan_change/baseline_makespan*100:+.1f}%)")
                    print(f"  Cost: ${cost_change:+,.2f} ({cost_change/baseline_cost*100:+.1f}%)")
                    print(f"{'='*60}\n")
                        
                    # Add scenario metadata
                    scenario_result['scenario_name'] = scenario_name
                    scenario_result['scenario_description'] = scenario.description
                    scenario_result['modifications'] = scenario.modifications
                    
                    results[scenario_name] = scenario_result
                    
                except Exception as e:
                    print(f"ERROR: {str(e)}")
                    print(f"{'='*60}\n")
                    results[scenario_name] = {
                        'status': 'ERROR',
                        'error': str(e),
                        'scenario_name': scenario_name,
                        'scenario_description': scenario.description
                    }
                    
        return results
    
    def create_results_summary_table(self, results: Dict[str, Dict]) -> pd.DataFrame:
        """Create a summary table of scenario results for easy comparison."""
        summary_data = []
        
        for scenario_name, result in results.items():
            if result.get('status') in ['OPTIMAL', 'FEASIBLE', 'INFEASIBLE']:
                row = {
                    'Scenario': scenario_name,
                    'Status': result.get('status', 'N/A'),
                    'Makespan (days)': result.get('makespan', 0),
                    'Total Cost ($)': result.get('total_cost', 0),
                }
                
                # Add change from baseline if not baseline
                if scenario_name != 'Baseline' and 'Baseline' in results:
                    baseline = results['Baseline']
                    makespan_change = result.get('makespan', 0) - baseline.get('makespan', 0)
                    cost_change = result.get('total_cost', 0) - baseline.get('total_cost', 0)
                    makespan_pct = (makespan_change / baseline.get('makespan', 1)) * 100
                    cost_pct = (cost_change / baseline.get('total_cost', 1)) * 100
                    
                    row['Makespan Change'] = f"{makespan_change:+.1f} days ({makespan_pct:+.1f}%)"
                    row['Cost Change'] = f"${cost_change:+,.0f} ({cost_pct:+.1f}%)"
                else:
                    row['Makespan Change'] = '-'
                    row['Cost Change'] = '-'
                
                summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def _run_ortools_optimization(self, data_processor: DataProcessor, **kwargs) -> Dict:
        """Run OR-Tools optimization on the given data processor."""
        scheduler_data = data_processor.preprocess_for_scheduler()
        scheduler = ConstructionScheduler(kwargs.get('project_horizon', 100))
        
        # Add tasks, resources, and constraints
        for task in scheduler_data['tasks'].values():
            scheduler.add_task(task)
        for resource in scheduler_data['resources'].values():
            scheduler.add_resource(resource)
        for weather_constraint in scheduler_data['weather_constraints']:
            scheduler.add_weather_constraint(weather_constraint)
        
        # Solve
        return scheduler.solve(
            time_limit_seconds=kwargs.get('time_limit', 60),
            objective_type=kwargs.get('objective_type', 'minimize_makespan')
        )
    
    def _run_genetic_optimization(self, data_processor: DataProcessor, **kwargs) -> Dict:
        """Run genetic algorithm optimization on the given data processor."""
        ga_data = data_processor.preprocess_for_genetic_algorithm()
        optimizer = ResourceOptimizer(
            population_size=kwargs.get('population_size', 50),
            max_generations=kwargs.get('max_generations', 200),
            mutation_rate=kwargs.get('mutation_rate', 0.1)
        )
        
        # Add tasks and resources
        for task_id, task_data in ga_data['tasks'].items():
            optimizer.add_task(task_id, **task_data)
        for resource_id, resource_data in ga_data['resources'].items():
            optimizer.add_resource(resource_id, **resource_data)
        
        return optimizer.optimize(verbose=False)
    
    def compare_scenarios(self, results: Dict[str, Dict]) -> pd.DataFrame:
        """Create a comparison table of scenario results."""
        comparison_data = []
        
        print(f"\n{'='*60}")
        print(f"COMPARING SCENARIOS")
        print(f"{'='*60}")
        print(f"Total scenarios in results: {len(results)}")
        
        for scenario_name, result in results.items():
            status = result.get('status', 'UNKNOWN')
            print(f"\nScenario: {scenario_name}")
            print(f"  Status: {status}")
            print(f"  Makespan: {result.get('makespan', 'N/A')}")
            print(f"  Total Cost: ${result.get('total_cost', 0):,.2f}")
            
            if status in ['OPTIMAL', 'FEASIBLE']:
                comparison_data.append({
                    'Scenario': scenario_name,
                    'Status': result['status'],
                    'Makespan (days)': result.get('makespan', 'N/A'),
                    'Total Cost': f"${result.get('total_cost', 0):.2f}",
                    'Solve Time (s)': result.get('solver_stats', {}).get('solve_time', 
                                               result.get('generations', 'N/A')),
                    'Description': result.get('scenario_description', '')
                })
                print(f"  ✓ Added to comparison table")
            else:
                comparison_data.append({
                    'Scenario': scenario_name,
                    'Status': result.get('status', 'ERROR'),
                    'Makespan (days)': 'N/A',
                    'Total Cost': 'N/A',
                    'Solve Time (s)': 'N/A',
                    'Description': result.get('scenario_description', result.get('error', ''))
                })
                print(f"  ✗ Status not OPTIMAL/FEASIBLE - showing as error")
        
        print(f"\nTotal rows in comparison table: {len(comparison_data)}")
        print(f"{'='*60}\n")
        
        return pd.DataFrame(comparison_data)
    
    def calculate_sensitivity_metrics(self, results: Dict[str, Dict]) -> Dict:
        """Calculate sensitivity metrics comparing scenarios to baseline."""
        if 'Baseline' not in results:
            return {}
            
        baseline = results['Baseline']
        baseline_makespan = baseline.get('makespan', 0)
        baseline_cost = baseline.get('total_cost', 0)
        
        sensitivity_metrics = {}
        
        for scenario_name, result in results.items():
            if scenario_name == 'Baseline' or result.get('status') not in ['OPTIMAL', 'FEASIBLE']:
                continue
                
            scenario_makespan = result.get('makespan', 0)
            scenario_cost = result.get('total_cost', 0)
            
            # Calculate percentage changes
            makespan_change = ((scenario_makespan - baseline_makespan) / baseline_makespan * 100) if baseline_makespan > 0 else 0
            cost_change = ((scenario_cost - baseline_cost) / baseline_cost * 100) if baseline_cost > 0 else 0
            
            sensitivity_metrics[scenario_name] = {
                'makespan_change_pct': makespan_change,
                'cost_change_pct': cost_change,
                'makespan_change_days': scenario_makespan - baseline_makespan,
                'cost_change_dollars': scenario_cost - baseline_cost
            }
            
        return sensitivity_metrics


def create_custom_scenario(name: str, description: str, modifications: List[Dict]) -> Scenario:
    """Helper function to create a custom scenario from modification specifications."""
    scenario = Scenario(name, description)
    
    for mod in modifications:
        modification_type = mod.pop('type')  # Remove type from dict
        scenario.add_modification(modification_type, **mod)
        
    return scenario


if __name__ == "__main__":
    # Test scenario analysis
    from ..utils.data_processor import create_sample_data
    
    # Create sample data and processor
    tasks_df, resources_df, weather_df = create_sample_data()
    dependencies = [('T1', 'T2'), ('T2', 'T3'), ('T3', 'T4'), ('T3', 'T5')]
    
    processor = DataProcessor()
    processor.load_from_dataframes(tasks_df, resources_df, weather_df, dependencies)
    
    # Create analyzer
    analyzer = ScenarioAnalyzer(processor)
    
    # Add predefined scenarios
    scenarios = analyzer.create_predefined_scenarios()
    for scenario in scenarios:
        analyzer.add_scenario(scenario)
    
    # Run analysis
    results = analyzer.run_scenario_analysis(['Labor Shortage', 'Weather Delays'], algorithm='ortools')
    
    # Display results
    comparison_df = analyzer.compare_scenarios(results)
    print("Scenario Comparison:")
    print(comparison_df.to_string(index=False))
    
    sensitivity = analyzer.calculate_sensitivity_metrics(results)
    print("\nSensitivity Analysis:")
    for scenario, metrics in sensitivity.items():
        print(f"{scenario}: Makespan {metrics['makespan_change_pct']:+.1f}%, Cost {metrics['cost_change_pct']:+.1f}%")