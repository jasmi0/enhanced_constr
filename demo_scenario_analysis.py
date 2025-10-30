"""
Scenario Analysis Demonstration

This script demonstrates the key capabilities of the scenario analysis system.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.utils.data_processor import DataProcessor, create_sample_data
from src.utils.scenario_analysis import ScenarioAnalyzer, Scenario, create_custom_scenario
from src.utils.scenario_visualization import ScenarioVisualizer
import pandas as pd

def demonstrate_scenario_analysis():
    """Demonstrate scenario analysis capabilities."""
    
    print(" AI-Enhanced Construction Scheduling - Scenario Analysis Demo")
    print("=" * 70)
    
    # Step 1: Setup base data
    print("\n Step 1: Loading Project Data")
    print("-" * 30)
    
    tasks_df, resources_df, weather_df = create_sample_data()
    dependencies = [('T1', 'T2'), ('T2', 'T3'), ('T3', 'T4'), ('T3', 'T5')]
    
    processor = DataProcessor()
    processor.load_from_dataframes(tasks_df, resources_df, weather_df, dependencies)
    
    print(f" Loaded {len(tasks_df)} tasks, {len(resources_df)} resources")
    
    # Step 2: Create scenario analyzer
    print("\n Step 2: Setting Up Scenario Analyzer")
    print("-" * 40)
    
    analyzer = ScenarioAnalyzer(processor)
    
    # Add predefined scenarios
    predefined_scenarios = analyzer.create_predefined_scenarios()
    for scenario in predefined_scenarios:
        analyzer.add_scenario(scenario)
    
    print(f" Created {len(predefined_scenarios)} predefined scenarios:")
    for scenario in predefined_scenarios:
        print(f"   • {scenario.name}: {scenario.description}")
    
    # Step 3: Run baseline optimization
    print("\n⚡ Step 3: Running Baseline Optimization")
    print("-" * 40)
    
    baseline_result = analyzer.run_baseline(algorithm='ortools')
    print(f" Baseline Status: {baseline_result['status']}")
    print(f" Baseline Makespan: {baseline_result['makespan']} days")
    print(f" Baseline Cost: ${baseline_result['total_cost']:,.2f}")
    
    # Step 4: Analyze selected scenarios
    print("\n Step 4: Scenario Analysis")
    print("-" * 30)
    
    test_scenarios = ['Labor Shortage', 'Labor Abundance', 'Material Cost Spike', 'Extended Weather Delays']
    
    print(f"Running analysis for scenarios: {', '.join(test_scenarios)}")
    results = analyzer.run_scenario_analysis(scenarios=test_scenarios, algorithm='ortools')
    
    # Step 5: Generate comparison
    print("\n Step 5: Results Comparison")
    print("-" * 30)
    
    comparison_df = analyzer.compare_scenarios(results)
    print("\nScenario Comparison Table:")
    print(comparison_df.to_string(index=False))
    
    # Step 6: Sensitivity analysis
    print("\n Step 6: Sensitivity Analysis")
    print("-" * 32)
    
    sensitivity_metrics = analyzer.calculate_sensitivity_metrics(results)
    
    print("\nSensitivity Metrics (vs Baseline):")
    for scenario, metrics in sensitivity_metrics.items():
        makespan_change = metrics['makespan_change_pct']
        cost_change = metrics['cost_change_pct']
        print(f"   • {scenario}:")
        print(f"     - Schedule Impact: {makespan_change:+.1f}%")
        print(f"     - Cost Impact: {cost_change:+.1f}%")
    
    # Step 7: Create custom scenario
    print("\n Step 7: Custom Scenario Creation")
    print("-" * 35)
    
    # Create a complex custom scenario
    crisis_scenario = Scenario(
        "Project Crisis",
        "Combined labor shortage, material cost spike, and weather delays"
    )
    
    # Add modifications one by one
    crisis_scenario.add_modification('change_labor_availability', multiplier=0.6)  # 40% labor reduction
    crisis_scenario.add_modification('change_material_cost', material='steel', multiplier=1.5)  # 50% steel cost increase
    crisis_scenario.add_modification('add_weather_delay', affected_tasks=['T1', 'T4'], 
                                   delay_periods=[[20, 25]], weather_type='storm')
    
    analyzer.add_scenario(crisis_scenario)
    
    # Run the crisis scenario
    crisis_results = analyzer.run_scenario_analysis(scenarios=['Project Crisis'], algorithm='ortools')
    
    if 'Project Crisis' in crisis_results and crisis_results['Project Crisis']['status'] in ['OPTIMAL', 'FEASIBLE']:
        crisis_result = crisis_results['Project Crisis']
        baseline_makespan = baseline_result['makespan']
        baseline_cost = baseline_result['total_cost']
        
        makespan_impact = ((crisis_result['makespan'] - baseline_makespan) / baseline_makespan) * 100
        cost_impact = ((crisis_result['total_cost'] - baseline_cost) / baseline_cost) * 100
        
        print(f" Crisis Scenario Results:")
        print(f"   • Status: {crisis_result['status']}")
        print(f"   • Makespan: {crisis_result['makespan']} days ({makespan_impact:+.1f}%)")
        print(f"   • Cost: ${crisis_result['total_cost']:,.2f} ({cost_impact:+.1f}%)")
    
    # Step 8: Summary insights
    print("\n Step 8: Key Insights")
    print("-" * 22)
    
    print("\nScenario Analysis Insights:")
    
    # Find most impactful scenarios
    if sensitivity_metrics:
        worst_schedule = max(sensitivity_metrics.items(), key=lambda x: x[1]['makespan_change_pct'])
        worst_cost = max(sensitivity_metrics.items(), key=lambda x: x[1]['cost_change_pct'])
        best_schedule = min(sensitivity_metrics.items(), key=lambda x: x[1]['makespan_change_pct'])
        
        print(f"   • Worst Schedule Impact: {worst_schedule[0]} ({worst_schedule[1]['makespan_change_pct']:+.1f}%)")
        print(f"   • Worst Cost Impact: {worst_cost[0]} ({worst_cost[1]['cost_change_pct']:+.1f}%)")
        print(f"   • Best Schedule Improvement: {best_schedule[0]} ({best_schedule[1]['makespan_change_pct']:+.1f}%)")
    
    # Risk assessment
    high_risk_scenarios = []
    for scenario, metrics in sensitivity_metrics.items():
        if metrics['makespan_change_pct'] > 10 or metrics['cost_change_pct'] > 10:
            high_risk_scenarios.append(scenario)
    
    if high_risk_scenarios:
        print(f"   • High Risk Scenarios: {', '.join(high_risk_scenarios)}")
    
    print("\n Visualization Components Available:")
    print("   • Scenario comparison charts")
    print("   • Sensitivity tornado diagrams")
    print("   • Risk assessment matrices")
    print("   • Gantt chart comparisons")
    print("   • Cost breakdown analysis")
    print("   • Comprehensive summary tables")
    
    print("\n" + "=" * 70)
    print(" Scenario Analysis Demo Complete!")
    print("\nNext Steps:")
    print("• Run 'streamlit run app.py' to access the interactive interface")
    print("• Navigate to 'Scenario Analysis' section")
    print("• Explore predefined scenarios or create custom ones")
    print("• Compare results and analyze sensitivities")

if __name__ == "__main__":
    demonstrate_scenario_analysis()