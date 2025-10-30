"""
Test scenario analysis functionality.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.utils.data_processor import DataProcessor, create_sample_data
from src.utils.scenario_analysis import ScenarioAnalyzer, Scenario
from src.utils.scenario_visualization import ScenarioVisualizer
import pandas as pd

def test_scenario_analysis():
    """Test the scenario analysis functionality."""
    print(" Testing Scenario Analysis...")
    
    # Create sample data
    tasks_df, resources_df, weather_df = create_sample_data()
    dependencies = [('T1', 'T2'), ('T2', 'T3'), ('T3', 'T4'), ('T3', 'T5')]
    
    # Create data processor
    processor = DataProcessor()
    processor.load_from_dataframes(tasks_df, resources_df, weather_df, dependencies)
    
    # Create scenario analyzer
    analyzer = ScenarioAnalyzer(processor)
    
    # Add predefined scenarios
    scenarios = analyzer.create_predefined_scenarios()
    for scenario in scenarios:
        analyzer.add_scenario(scenario)
    
    print(f" Created {len(scenarios)} predefined scenarios")
    
    # Run scenario analysis with a subset of scenarios
    test_scenarios = ['Labor Shortage', 'Weather Delays']
    results = analyzer.run_scenario_analysis(scenarios=test_scenarios, algorithm='ortools')
    
    print(f" Analyzed {len(results)} scenarios")
    
    # Test comparison and sensitivity analysis
    comparison_df = analyzer.compare_scenarios(results)
    sensitivity_metrics = analyzer.calculate_sensitivity_metrics(results)
    
    print(f" Generated comparison table with {len(comparison_df)} rows")
    print(f" Calculated sensitivity metrics for {len(sensitivity_metrics)} scenarios")
    
    # Test visualization components
    visualizer = ScenarioVisualizer()
    
    # Test each visualization type
    fig1 = visualizer.create_scenario_comparison_chart(comparison_df, 'Makespan (days)')
    print(" Created scenario comparison chart")
    
    if sensitivity_metrics:
        fig2 = visualizer.create_sensitivity_tornado_chart(sensitivity_metrics)
        print(" Created sensitivity tornado chart")
        
        fig3 = visualizer.create_risk_assessment_matrix(results)
        print(" Created risk assessment matrix")
    
    fig4 = visualizer.create_cost_breakdown_comparison(results)
    print(" Created cost breakdown comparison")
    
    fig5 = visualizer.create_scenario_summary_table(comparison_df, sensitivity_metrics)
    print(" Created scenario summary table")
    
    # Test custom scenario creation
    custom_scenario = Scenario("Test Custom", "Custom test scenario")
    custom_scenario.add_modification('change_labor_availability', multiplier=0.8)
    analyzer.add_scenario(custom_scenario)
    
    print(" Created and added custom scenario")
    
    return True

if __name__ == "__main__":
    print("🏗️ Scenario Analysis Test Suite")
    print("=" * 50)
    
    try:
        test_scenario_analysis()
        print("=" * 50)
        print(" All scenario analysis tests passed!")
    except Exception as e:
        print(f" Test failed: {e}")
        import traceback
        traceback.print_exc()