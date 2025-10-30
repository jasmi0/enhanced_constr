# Scenario Analysis Documentation

## Overview

The scenario analysis module provides comprehensive what-if analysis capabilities for construction project scheduling. It allows users to test different scenarios, compare outcomes, and assess risks before making project decisions.

## Features

### 1. Predefined Scenarios

The system includes six predefined scenarios that cover common construction project challenges:

- **Labor Shortage**: Reduced labor availability by 30%
- **Labor Abundance**: Increased labor availability by 50%
- **Material Cost Spike**: Steel costs increased by 100%
- **Extended Weather Delays**: Additional rain periods affecting outdoor tasks
- **Rush Project**: All task durations reduced by 20% (crash schedule)
- **Priority Rebalancing**: Roofing task set to highest priority

### 2. Custom Scenario Creation

Users can create custom scenarios by combining multiple modifications:

#### Available Modification Types:

1. **Labor Availability Changes**
   - Multiply total labor capacity by a factor (0.1 to 3.0)
   - Useful for testing workforce changes

2. **Material Cost Modifications**
   - Change cost per unit for specific materials
   - Test impact of market price fluctuations

3. **Weather Delays**
   - Add blocked periods for weather-sensitive tasks
   - Specify affected tasks and delay periods

4. **Task Duration Changes**
   - Modify duration of specific tasks
   - Test acceleration or delay scenarios

5. **Resource Constraints**
   - Add or modify resource capacity limits
   - Test equipment or material availability

### 3. Analysis Capabilities

#### Scenario Comparison
- Side-by-side comparison of key metrics
- Makespan, cost, and status comparison
- Solve time and performance metrics

#### Sensitivity Analysis
- Percentage change from baseline
- Tornado charts showing impact magnitude
- Risk assessment matrices

#### Visualization Components
- Gantt chart comparisons
- Cost breakdown analysis
- Resource utilization charts
- Risk assessment matrices
- Interactive summary tables

## Usage

### 1. Via Streamlit App

1. Navigate to the "Scenario Analysis" section
2. Select predefined scenarios or create custom ones
3. Run analysis with chosen algorithm (OR-Tools or Genetic)
4. Review results in interactive visualizations

### 2. Via Python API

```python
from src.utils.scenario_analysis import ScenarioAnalyzer, Scenario
from src.utils.data_processor import DataProcessor

# Setup
processor = DataProcessor()
# ... load your data ...

analyzer = ScenarioAnalyzer(processor)

# Add predefined scenarios
scenarios = analyzer.create_predefined_scenarios()
for scenario in scenarios:
    analyzer.add_scenario(scenario)

# Create custom scenario
custom = Scenario("My Scenario", "Description")
custom.add_modification('change_labor_availability', multiplier=0.8)
analyzer.add_scenario(custom)

# Run analysis
results = analyzer.run_scenario_analysis(['My Scenario'], algorithm='ortools')

# Get comparison
comparison_df = analyzer.compare_scenarios(results)
sensitivity = analyzer.calculate_sensitivity_metrics(results)
```

### 3. Custom Scenario Creation Example

```python
# Create a crisis scenario
crisis = Scenario("Project Crisis", "Multiple challenges combined")

# Add labor shortage
crisis.add_modification('change_labor_availability', multiplier=0.6)

# Add material cost increase
crisis.add_modification('change_material_cost', material='steel', multiplier=1.5)

# Add weather delays
crisis.add_modification('add_weather_delay', 
                       affected_tasks=['T1', 'T4'],
                       delay_periods=[[15, 20], [30, 35]],
                       weather_type='storms')

analyzer.add_scenario(crisis)
```

## Interpretation Guide

### Key Metrics

1. **Status**: OPTIMAL, FEASIBLE, or INFEASIBLE
2. **Makespan**: Project completion time in days
3. **Total Cost**: Combined labor and material costs
4. **Solve Time**: Optimization algorithm execution time

### Sensitivity Analysis

- **Positive % Change**: Scenario performs worse than baseline
- **Negative % Change**: Scenario performs better than baseline
- **High Sensitivity**: Large percentage changes indicate critical factors

### Risk Assessment

The risk matrix plots scenarios by cost vs schedule impact:
- **Green Zone**: Low risk (both metrics improved)
- **Orange Zone**: Medium risk (one metric improved, one worse)
- **Red Zone**: High risk (both metrics worse)

## Best Practices

### 1. Scenario Design
- Start with single-factor scenarios
- Gradually combine factors for complex scenarios
- Test realistic ranges for modifications

### 2. Analysis Workflow
1. Run baseline optimization first
2. Test individual risk factors
3. Combine factors for stress testing
4. Analyze sensitivity patterns
5. Develop mitigation strategies

### 3. Interpretation
- Focus on relative changes, not absolute values
- Consider both schedule and cost impacts
- Look for threshold effects and non-linear responses
- Validate insights with domain expertise

## Technical Details

### Algorithms Supported
- **OR-Tools CP-SAT**: Constraint-based optimization
- **Genetic Algorithm**: Evolutionary optimization

### Performance
- Typical analysis time: 1-5 minutes for multiple scenarios
- Memory usage: Scales with number of tasks and scenarios
- Parallel execution: Scenarios run sequentially

### Limitations
- Weather constraints may cause infeasible solutions
- Large parameter changes may exceed solver limits
- Visualization limited to successful scenarios

## Examples

See the following files for examples:
- `demo_scenario_analysis.py`: Complete demonstration
- `test_scenario_analysis.py`: Testing examples
- `app.py`: Interactive interface implementation

## Extension Points

The scenario analysis system is designed for extension:

1. **New Modification Types**: Add to `_apply_modification()` method
2. **Custom Visualizations**: Extend `ScenarioVisualizer` class
3. **Additional Metrics**: Enhance comparison and sensitivity calculations
4. **Export Formats**: Add new output formats for results

## Troubleshooting

### Common Issues

1. **Infeasible Solutions**
   - Reduce constraint severity
   - Check resource availability
   - Verify dependency constraints

2. **Slow Performance**
   - Reduce time limits
   - Limit number of scenarios
   - Use genetic algorithm for speed

3. **Visualization Errors**
   - Ensure successful scenario results
   - Check data format consistency
   - Update Plotly version if needed

### Debug Mode

Enable detailed logging by setting debug flags in the analyzer:

```python
analyzer.debug = True
results = analyzer.run_scenario_analysis(...)
```

## Future Enhancements

Planned features for future versions:

1. **Monte Carlo Simulation**: Statistical uncertainty analysis
2. **Multi-Objective Optimization**: Pareto frontier analysis
3. **Real-time Updates**: Live scenario modification
4. **Machine Learning**: Predictive scenario outcomes
5. **Integration**: External data sources and APIs