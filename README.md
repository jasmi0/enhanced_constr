# AI-Enhanced Construction Scheduling & Resource Allocation

## Overview

This project uses AI to optimize construction scheduling and resource allocation. By considering factors like available labor, materials, weather, and deadlines, it provides optimized schedules with an interactive dashboard for visualization and real-time adjustments.

## Features

- **Constraint-based Scheduling**: Uses OR-Tools CP-SAT solver for optimal task scheduling
- **Resource Optimization**: Genetic algorithms for dynamic resource allocation
- **Weather Integration**: Accounts for weather constraints affecting project timelines
- **Interactive Dashboard**: Streamlit-based UI with real-time visualization
- **Multiple Algorithms**: Compare OR-Tools and Genetic Algorithm results
- **Comprehensive Scenario Analysis**: What-if studies with predefined and custom scenarios
- **Risk Assessment**: Sensitivity analysis and risk matrices for decision support
- **Export Capabilities**: Download results in CSV and JSON formats
- **Advanced Visualizations**: Gantt charts, resource utilization, and comparative analysis

## Technologies Used

- **Language**: Python 3.8+
- **Optimization**: OR-Tools (CP-SAT solver), Genetic Algorithms
- **Frontend/UI**: Streamlit for user interaction and visualization
- **Data Processing**: pandas, NumPy
- **Visualization**: Plotly for Gantt charts and resource allocation graphs

## Installation

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Setup

1. **Clone the repository** (or ensure you're in the project directory):
   ```bash
   cd /home/alerman/projects/bena_projects/enhanced_constr
   ```

2. **Activate virtual environment**:
   ```bash
   source venv/bin/activate
   ```

3. **Install dependencies** (already done):
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Access the application**:
   - Open your browser and go to `http://localhost:8501`
   - The app will automatically open if your browser supports it

### Using the Application

#### 1. Data Input
- **Upload CSV Files**: Upload your own project data files
- **Use Sample Data**: Click "Load Sample Data" to use the provided example
- **Manual Entry**: (Coming soon) Enter data directly in the interface

#### 2. Required Data Format

**Tasks CSV Format** (`sample_tasks.csv`):
```csv
task_id,task_name,duration,labor_required,materials_needed,earliest_start,latest_finish,priority
T1,Site Preparation,5,3,"{""cement"": 10}",0,10,1.0
```

**Resources CSV Format** (`sample_resources.csv`):
```csv
resource_id,resource_type,capacity,cost_per_unit
workers,labor,10,100
cement,material,100,1
```

**Weather CSV Format** (`sample_weather.csv`):
```csv
weather_type,affected_tasks,blocked_periods
rain,"T4,T7","[[15, 20], [35, 40]]"
```

**Dependencies CSV Format** (`sample_dependencies.csv`):
```csv
predecessor,successor
T1,T2
T2,T3
```

#### 3. Optimization
- Choose between OR-Tools CP-SAT Solver, Genetic Algorithm, or both
- Configure parameters like project horizon, time limits
- Set optimization objectives (minimize duration, cost, etc.)
- Run optimization and view real-time progress

#### 4. Results Analysis
- **Gantt Charts**: Visual project timelines
- **Resource Utilization**: Track resource usage over time
- **Cost Analysis**: Resource allocation and cost breakdown
- **Performance Metrics**: Algorithm performance comparison

#### 5. Scenario Analysis
- **Predefined Scenarios**: Labor shortages, material cost spikes, weather delays, rush projects
- **Custom Scenarios**: Create complex what-if scenarios with multiple modifications
- **Comparative Analysis**: Side-by-side comparison of scenario outcomes
- **Sensitivity Analysis**: Tornado charts showing impact magnitude
- **Risk Assessment**: Risk matrices plotting cost vs schedule impact
- **Interactive Visualizations**: Gantt comparisons, cost breakdowns, summary tables

#### 6. Export & Reporting
- Download results as CSV files
- Export scenario comparison reports
- Save custom scenarios for future analysis

## Project Structure

```
enhanced_constr/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                      # This file
├── demo_scenario_analysis.py      # Scenario analysis demonstration
├── test_scenario_analysis.py      # Scenario analysis testing
├── data/
│   └── sample/                    # Sample CSV files
│       ├── sample_tasks.csv
│       ├── sample_resources.csv
│       ├── sample_weather.csv
│       └── sample_dependencies.csv
├── docs/
│   └── SCENARIO_ANALYSIS.md       # Detailed scenario analysis documentation
├── src/
│   ├── models/
│   │   ├── scheduler.py           # OR-Tools scheduling engine
│   │   └── genetic_optimizer.py  # Genetic algorithm optimizer
│   ├── utils/
│   │   ├── data_processor.py     # Data validation and preprocessing
│   │   ├── scenario_analysis.py  # Scenario analysis engine
│   │   └── scenario_visualization.py  # Scenario visualization components
│   └── visualization/
│       └── charts.py              # Plotly visualization components
└── venv/                          # Virtual environment
```

## Algorithm Details

### OR-Tools CP-SAT Solver
- **Purpose**: Constraint-based scheduling optimization
- **Features**: 
  - Task dependencies and precedence constraints
  - Resource capacity constraints
  - Weather and availability constraints
  - Multiple optimization objectives
- **Best for**: Projects with complex constraints and dependencies

### Genetic Algorithm
- **Purpose**: Resource allocation optimization
- **Features**:
  - Population-based evolutionary optimization
  - Multi-objective fitness function
  - Dynamic resource allocation
  - Handles resource costs and priorities
- **Best for**: Projects focused on resource efficiency and cost optimization

## Data Schema

### Tasks
- `task_id`: Unique identifier for the task
- `task_name`: Human-readable task name
- `duration`: Task duration in days
- `labor_required`: Number of workers needed
- `materials_needed`: JSON string of materials and quantities
- `earliest_start`: Earliest start day (optional)
- `latest_finish`: Latest finish day (optional)
- `priority`: Task priority (0.0 to 1.0)

### Resources
- `resource_id`: Unique identifier for the resource
- `resource_type`: Type of resource (labor, material, equipment)
- `capacity`: Total available capacity
- `cost_per_unit`: Cost per unit of resource (optional)
- `availability_periods`: JSON array of available time periods (optional)

### Weather Constraints
- `weather_type`: Type of weather condition
- `affected_tasks`: Comma-separated list of affected task IDs
- `blocked_periods`: JSON array of blocked time periods

### Dependencies
- `predecessor`: Task that must complete first
- `successor`: Task that depends on the predecessor

## Scenario Analysis

The scenario analysis system provides comprehensive what-if analysis capabilities:

### Predefined Scenarios
- **Labor Shortage**: Tests impact of reduced workforce availability
- **Labor Abundance**: Tests benefits of increased workforce capacity
- **Material Cost Spike**: Analyzes impact of supply chain cost increases
- **Extended Weather Delays**: Tests resilience to prolonged weather issues
- **Rush Project**: Evaluates crash schedule feasibility
- **Priority Rebalancing**: Tests impact of changing task priorities

### Custom Scenarios
Create complex scenarios by combining:
- Labor availability changes (0.1x to 3.0x multipliers)
- Material cost modifications for specific materials
- Weather delay periods for outdoor tasks
- Task duration adjustments
- Resource constraint modifications

### Analysis Features
- **Sensitivity Analysis**: Tornado charts showing impact magnitude
- **Risk Assessment**: Cost vs schedule impact matrices
- **Comparative Visualizations**: Side-by-side Gantt charts and metrics
- **Export Capabilities**: Download scenario comparison results

For detailed documentation, see `docs/SCENARIO_ANALYSIS.md`

## Example Workflow

1. **Load Data**: Use sample data or upload your CSV files
2. **Configure Optimization**: Choose algorithm and set parameters
3. **Run Optimization**: Execute the scheduling algorithm
4. **Analyze Results**: Review Gantt charts, resource utilization, costs
5. **Scenario Analysis**: Test predefined scenarios or create custom ones
6. **Risk Assessment**: Review sensitivity analysis and risk matrices
7. **Export Results**: Download optimized schedules and scenario comparisons

## Results and Performance

The system provides:
- **Optimized Schedules**: Minimize project completion time while respecting constraints
- **Resource Efficiency**: Optimal allocation of labor, materials, and equipment
- **Cost Analysis**: Detailed breakdown of project costs
- **Risk Assessment**: Impact analysis of weather and resource constraints
- **Real-time Updates**: Interactive scenario simulation capabilities

## Troubleshooting

### Common Issues

1. **ImportError when running the app**:
   ```bash
   # Make sure you're in the project directory and virtual environment is activated
   source venv/bin/activate
   cd /home/alerman/projects/bena_projects/enhanced_constr
   ```

2. **Data validation errors**:
   - Check CSV file formats match the expected schema
   - Ensure all required columns are present
   - Verify data types (numbers for duration, labor, etc.)

3. **Optimization failures**:
   - Reduce project horizon or time limits
   - Check for circular dependencies
   - Ensure resource capacities are sufficient

4. **Performance issues**:
   - Reduce population size for genetic algorithm
   - Decrease max generations or time limits
   - Simplify the problem by reducing tasks or constraints

### Getting Help

If you encounter issues:
1. Check the error messages in the Streamlit interface
2. Verify your data format matches the sample files
3. Try using the sample data first to ensure the system works
4. Check the terminal/console for detailed error messages

## Future Improvements

- **Real-time Data Integration**: Connect to external APIs for weather, resource prices
- **Machine Learning**: Predictive analytics for task duration and resource needs
- **Multi-project Coordination**: Optimize across multiple concurrent projects
- **Mobile Interface**: Responsive design for mobile devices
- **Collaboration Features**: Multi-user project management capabilities
- **Advanced Reporting**: Comprehensive project reports and analytics
