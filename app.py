import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import io
import json
import plotly.graph_objects as go

sys.path.append(str(Path(__file__).parent / 'src'))

from src.models.scheduler import ConstructionScheduler, Task, Resource, WeatherConstraint
from src.models.genetic_optimizer import ResourceOptimizer
from src.utils.data_processor import DataProcessor, create_sample_data
from src.visualization.charts import ScheduleVisualizer
from src.utils.scenario_analysis import ScenarioAnalyzer, Scenario, create_custom_scenario
from src.utils.scenario_visualization import ScenarioVisualizer


def initialize_session_state():
    if 'data_processor' not in st.session_state:
        st.session_state.data_processor = DataProcessor()
    if 'optimization_results' not in st.session_state:
        st.session_state.optimization_results = None
    if 'scenario_analyzer' not in st.session_state:
        st.session_state.scenario_analyzer = None
    if 'scenario_results' not in st.session_state:
        st.session_state.scenario_results = None
    if 'sample_data_loaded' not in st.session_state:
        st.session_state.sample_data_loaded = False


def load_sample_data():
    try:
        tasks_df, resources_df, weather_df = create_sample_data()
        dependencies = [('T1', 'T2'), ('T2', 'T3'), ('T3', 'T4'), ('T3', 'T5')]

        success = st.session_state.data_processor.load_from_dataframes(
            tasks_df, resources_df, weather_df, dependencies
        )

        if success:
            st.session_state.sample_data_loaded = True
            return tasks_df, resources_df, weather_df, dependencies
        else:
            st.error("Failed to load sample data")
            return None, None, None, None
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        return None, None, None, None
def data_input_section():
    st.header("Data Input")

    input_method = st.radio(
        "Choose data input method:",
        ["Upload CSV Files", "Use Sample Data", "Manual Entry"]
    )

    if input_method == "Upload CSV Files":
        col1, col2 = st.columns(2)

        with col1:
            tasks_file = st.file_uploader("Upload Tasks CSV", type=['csv'])
            resources_file = st.file_uploader("Upload Resources CSV", type=['csv'])

        with col2:
            weather_file = st.file_uploader("Upload Weather CSV (Optional)", type=['csv'])
            dependencies_file = st.file_uploader("Upload Dependencies CSV (Optional)", type=['csv'])

        if tasks_file is not None:
            try:
                tasks_df = pd.read_csv(tasks_file)
                st.success(f"Tasks loaded: {len(tasks_df)} tasks")

                resources_df = None
                if resources_file is not None:
                    resources_df = pd.read_csv(resources_file)
                    st.success(f"Resources loaded: {len(resources_df)} resources")

                weather_df = None
                if weather_file is not None:
                    weather_df = pd.read_csv(weather_file)
                    st.success(f"Weather data loaded: {len(weather_df)} constraints")

                dependencies = []
                if dependencies_file is not None:
                    deps_df = pd.read_csv(dependencies_file)
                    dependencies = list(zip(deps_df['predecessor'], deps_df['successor']))
                    st.success(f"Dependencies loaded: {len(dependencies)} dependencies")

                success = st.session_state.data_processor.load_from_dataframes(
                    tasks_df, resources_df, weather_df, dependencies
                )

                if success:
                    st.success("All data loaded successfully!")
                    return True
                else:
                    st.error("Failed to validate and load data")
                    return False

            except Exception as e:
                st.error(f"Error loading files: {e}")
                return False

    elif input_method == "Use Sample Data":
        if st.button("Load Sample Data"):
            tasks_df, resources_df, weather_df, dependencies = load_sample_data()
            if tasks_df is not None:
                st.success("Sample data loaded successfully!")

                with st.expander("Sample Data Preview"):
                    st.subheader("Tasks")
                    st.dataframe(tasks_df)
                    st.subheader("Resources")
                    st.dataframe(resources_df)
                    st.subheader("Weather Constraints")
                    st.dataframe(weather_df)

                return True
            else:
                return False

        return st.session_state.sample_data_loaded

    elif input_method == "Manual Entry":
        st.info("Manual data entry feature coming soon. Please use CSV upload or sample data.")
        return False

    return False


def optimization_section():
    st.header("Optimization Configuration")

    algorithm = st.selectbox(
        "Choose optimization algorithm:",
        ["OR-Tools CP-SAT Solver", "Genetic Algorithm", "Both (Compare Results)"]
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        project_horizon = st.number_input("Project Horizon (days)", min_value=30, max_value=1000, value=100)

    with col2:
        time_limit = st.number_input("Time Limit (seconds)", min_value=10, max_value=600, value=60)

    with col3:
        objective = st.selectbox("Optimization Objective",
                                ["Minimize Project Duration", "Minimize Total Cost", "Balanced"])

    if algorithm in ["Genetic Algorithm", "Both (Compare Results)"]:
        st.subheader("Genetic Algorithm Parameters")
        ga_col1, ga_col2, ga_col3 = st.columns(3)

        with ga_col1:
            population_size = st.number_input("Population Size", min_value=20, max_value=200, value=50)

        with ga_col2:
            max_generations = st.number_input("Max Generations", min_value=50, max_value=1000, value=200)

        with ga_col3:
            mutation_rate = st.slider("Mutation Rate", min_value=0.01, max_value=0.5, value=0.1)

    if st.button("Run Optimization", type="primary"):
        if st.session_state.data_processor.tasks_df is None:
            st.error("Please load data first!")
            return

        with st.spinner("Running optimization..."):
            results = {}

            try:
                if algorithm in ["OR-Tools CP-SAT Solver", "Both (Compare Results)"]:
                    st.info("Running OR-Tools CP-SAT Solver...")

                    scheduler_data = st.session_state.data_processor.preprocess_for_scheduler()
                    scheduler = ConstructionScheduler(project_horizon)

                    for task in scheduler_data['tasks'].values():
                        scheduler.add_task(task)
                    for resource in scheduler_data['resources'].values():
                        scheduler.add_resource(resource)
                    for weather_constraint in scheduler_data['weather_constraints']:
                        scheduler.add_weather_constraint(weather_constraint)

                    objective_type = "minimize_makespan"
                    if objective == "Minimize Project Duration":
                        objective_type = "minimize_makespan"
                    elif objective == "Minimize Total Cost":
                        objective_type = "minimize_total_duration"

                    ortools_result = scheduler.solve(time_limit, objective_type)
                    results['OR-Tools'] = ortools_result

                if algorithm in ["Genetic Algorithm", "Both (Compare Results)"]:
                    st.info("Running Genetic Algorithm...")

                    ga_data = st.session_state.data_processor.preprocess_for_genetic_algorithm()
                    optimizer = ResourceOptimizer(
                        population_size=population_size,
                        max_generations=max_generations,
                        mutation_rate=mutation_rate
                    )

                    for task_id, task_data in ga_data['tasks'].items():
                        optimizer.add_task(task_id, **task_data)
                    for resource_id, resource_data in ga_data['resources'].items():
                        optimizer.add_resource(resource_id, **resource_data)

                    ga_result = optimizer.optimize(verbose=False)
                    results['Genetic Algorithm'] = ga_result

                st.session_state.optimization_results = results
                st.success("Optimization completed successfully!")

            except Exception as e:
                st.error(f"Optimization failed: {e}")
                st.exception(e)


def results_section():
    if st.session_state.optimization_results is None:
        st.info("Run optimization to see results here.")
        return

    st.header("Optimization Results")

    results = st.session_state.optimization_results

    st.subheader("Summary")
    summary_data = []

    for algorithm, result in results.items():
        summary_data.append({
            'Algorithm': algorithm,
            'Status': result.get('status', 'Unknown'),
            'Makespan (days)': result.get('makespan', 'N/A'),
            'Total Cost': f"${result.get('total_cost', 0):.2f}" if 'total_cost' in result else 'N/A',
            'Solve Time': f"{result.get('solver_stats', {}).get('solve_time', 0):.2f}s" if 'solver_stats' in result else 'N/A'
        })

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Gantt Chart", "Resource Usage", "Resource Allocation", "Performance"])

    visualizer = ScheduleVisualizer()

    with tab1:
        st.subheader("Project Schedule - Gantt Chart")

        for algorithm, result in results.items():
            if 'schedule' in result or 'allocation' in result:
                schedule_data = result.get('schedule', result.get('allocation', {}))

                if schedule_data:
                    st.write(f"**{algorithm}**")
                    gantt_fig = visualizer.create_gantt_chart(schedule_data, f"{algorithm} - Project Schedule")
                    st.plotly_chart(gantt_fig, use_container_width=True)

    with tab2:
        st.subheader("Resource Utilization Over Time")

        for algorithm, result in results.items():
            if 'schedule' in result or 'allocation' in result:
                schedule_data = result.get('schedule', result.get('allocation', {}))

                resources_info = None
                if hasattr(st.session_state.data_processor, 'resources_df') and st.session_state.data_processor.resources_df is not None:
                    resources_info = {}
                    for _, row in st.session_state.data_processor.resources_df.iterrows():
                        resources_info[row['resource_id']] = {
                            'resource_type': row['resource_type'],
                            'capacity': row['capacity'],
                            'total_capacity': row['capacity']
                        }

                if schedule_data:
                    st.write(f"**{algorithm}**")
                    resource_fig = visualizer.create_resource_utilization_chart(schedule_data, resources_info)
                    st.plotly_chart(resource_fig, use_container_width=True)

    with tab3:
        st.subheader("Resource Allocation by Task")
        
        # Add comparison chart if multiple algorithms
        if len(results) > 1:
            st.write("**Resource Type Comparison**")
            resource_comparison_fig = visualizer.create_resource_breakdown_comparison(results)
            st.plotly_chart(resource_comparison_fig, use_container_width=True)
            st.markdown("---")

        # Individual algorithm resource allocation
        for algorithm, result in results.items():
            if 'schedule' in result or 'allocation' in result:
                allocation_data = result.get('schedule', result.get('allocation', {}))

                if allocation_data:
                    st.write(f"**{algorithm}**")
                    pie_fig = visualizer.create_resource_allocation_pie(allocation_data)
                    st.plotly_chart(pie_fig, use_container_width=True)

    with tab4:
        st.subheader("Performance Dashboard")
        
        # Add algorithm comparison chart at the top
        if len(results) > 1:
            st.write("**Algorithm Comparison**")
            comparison_fig = visualizer.create_algorithm_comparison_chart(results)
            st.plotly_chart(comparison_fig, use_container_width=True)
            st.markdown("---")

        # Individual algorithm dashboards
        for algorithm, result in results.items():
            if result:
                st.write(f"**{algorithm} - Detailed Metrics**")
                dashboard_fig = visualizer.create_performance_dashboard(result)
                st.plotly_chart(dashboard_fig, use_container_width=True)
    with st.expander("Detailed Results"):
        for algorithm, result in results.items():
            st.subheader(f"{algorithm} - Detailed Results")

            if 'schedule' in result:
                schedule_df = pd.DataFrame.from_dict(result['schedule'], orient='index')
                st.dataframe(schedule_df)
            elif 'allocation' in result:
                allocation_list = []
                for task_id, task_info in result['allocation'].items():
                    row = {'task_id': task_id}
                    row.update(task_info)
                    allocation_list.append(row)
                allocation_df = pd.DataFrame(allocation_list)
                st.dataframe(allocation_df)

    st.subheader("Export Results")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Download Results as JSON"):
            json_str = json.dumps(results, indent=2, default=str)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name="optimization_results.json",
                mime="application/json"
            )

    with col2:
        if st.button("Download Schedule as CSV"):
            for algorithm, result in results.items():
                if 'schedule' in result:
                    schedule_df = pd.DataFrame.from_dict(result['schedule'], orient='index')
                    csv_str = schedule_df.to_csv(index=True)
                    st.download_button(
                        label=f"Download {algorithm} CSV",
                        data=csv_str,
                        file_name=f"{algorithm.lower().replace(' ', '_')}_schedule.csv",
                        mime="text/csv"
                    )


def scenario_analysis_section():
    st.header("Scenario Analysis")

    if st.session_state.data_processor.tasks_df is None:
        st.info("Please load data first from the Data Input section.")
        return

    if st.session_state.scenario_analyzer is None:
        st.session_state.scenario_analyzer = ScenarioAnalyzer(st.session_state.data_processor)

        predefined_scenarios = st.session_state.scenario_analyzer.create_predefined_scenarios()
        for scenario in predefined_scenarios:
            st.session_state.scenario_analyzer.add_scenario(scenario)

    tab1, tab2, tab3, tab4 = st.tabs(["Scenario Setup", "Quick Analysis", "Detailed Results", "Custom Scenarios"])

    with tab1:
        st.subheader("Available Scenarios")

        scenarios_info = []
        for name, scenario in st.session_state.scenario_analyzer.scenarios.items():
            scenarios_info.append({
                'Scenario': name,
                'Description': scenario.description,
                'Modifications': len(scenario.modifications)
            })

        if scenarios_info:
            scenarios_df = pd.DataFrame(scenarios_info)
            st.dataframe(scenarios_df, use_container_width=True)

        st.subheader("Select Scenarios to Analyze")
        available_scenarios = list(st.session_state.scenario_analyzer.scenarios.keys())
        selected_scenarios = st.multiselect(
            "Choose scenarios to compare:",
            available_scenarios,
            default=available_scenarios[:3] if len(available_scenarios) >= 3 else available_scenarios
        )

        col1, col2 = st.columns(2)
        with col1:
            algorithm = st.selectbox("Optimization Algorithm", ["OR-Tools", "Genetic Algorithm"])
        with col2:
            time_limit = st.number_input("Time Limit (seconds)", 10, 300, 60)
    
    with tab2:
        st.subheader("Quick Scenario Analysis")
        
        if selected_scenarios:
            if st.button(" Run Quick Analysis", type="primary"):
                with st.spinner("Running scenario analysis..."):
                   
                    algo = 'ortools' if algorithm == 'OR-Tools' else 'genetic'
                    
                    # Create expander for debug output
                    with st.expander("📋 Analysis Debug Log", expanded=False):
                        import io
                        import sys
                        
                        # Capture print output
                        old_stdout = sys.stdout
                        sys.stdout = buffer = io.StringIO()
                        
                        results = st.session_state.scenario_analyzer.run_scenario_analysis(
                            scenarios=selected_scenarios,
                            algorithm=algo,
                            time_limit=time_limit
                        )
                        
                        # Get the output and restore stdout
                        output = buffer.getvalue()
                        sys.stdout = old_stdout
                        
                        # Display the debug output
                        st.code(output, language='text')
                    
                    st.session_state.scenario_results = results
                    
                    # Show summary table
                    st.subheader(" Results Summary")
                    summary_df = st.session_state.scenario_analyzer.create_results_summary_table(results)
                    st.dataframe(summary_df, use_container_width=True)
                    
                   
                    comparison_df = st.session_state.scenario_analyzer.compare_scenarios(results)
                    
                    st.subheader(" Quick Comparison")
                    st.dataframe(comparison_df, use_container_width=True)
                    
                   
                    scenario_viz = ScenarioVisualizer()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_makespan = scenario_viz.create_scenario_comparison_chart(
                            comparison_df, 'Makespan (days)'
                        )
                        st.plotly_chart(fig_makespan, use_container_width=True)
                    
                    with col2:
                        fig_cost = scenario_viz.create_scenario_comparison_chart(
                            comparison_df, 'Total Cost'
                        )
                        st.plotly_chart(fig_cost, use_container_width=True)
        else:
            st.info("Please select scenarios to analyze in the Scenario Setup tab.")
    
    with tab3:
        st.subheader("Detailed Analysis Results")
        
        if st.session_state.scenario_results is not None:
            results = st.session_state.scenario_results
            scenario_viz = ScenarioVisualizer()
            
           
            sensitivity_metrics = st.session_state.scenario_analyzer.calculate_sensitivity_metrics(results)
            
            if sensitivity_metrics:
                st.subheader("🌪 Sensitivity Analysis")
                fig_tornado = scenario_viz.create_sensitivity_tornado_chart(sensitivity_metrics)
                st.plotly_chart(fig_tornado, use_container_width=True)
                
               
                st.subheader("⚠ Risk Assessment Matrix")
                fig_risk = scenario_viz.create_risk_assessment_matrix(results)
                st.plotly_chart(fig_risk, use_container_width=True)
            
           
            if len([s for s in results.keys() if results[s].get('status') in ['OPTIMAL', 'FEASIBLE']]) > 1:
                st.subheader(" Schedule Comparison")
                successful_scenarios = [
                    name for name, result in results.items() 
                    if result.get('status') in ['OPTIMAL', 'FEASIBLE']
                ][:3] 
                
                fig_gantt = scenario_viz.create_scenario_gantt_comparison(
                    results, successful_scenarios
                )
                st.plotly_chart(fig_gantt, use_container_width=True)
            
           
            st.subheader(" Cost Analysis")
            fig_cost_breakdown = scenario_viz.create_cost_breakdown_comparison(results)
            st.plotly_chart(fig_cost_breakdown, use_container_width=True)
            
           
            st.subheader(" Comprehensive Summary")
            comparison_df = st.session_state.scenario_analyzer.compare_scenarios(results)
            
            if len(comparison_df) > 0:
                fig_table = scenario_viz.create_scenario_summary_table(comparison_df, sensitivity_metrics)
                st.plotly_chart(fig_table, use_container_width=True)
            else:
                st.warning("No scenario data available for comparison. Results may have failed or be incomplete.")
                st.write("Debug - Results keys:", list(results.keys()))
                for scenario, result in results.items():
                    st.write(f"{scenario}: status={result.get('status')}, makespan={result.get('makespan')}, cost={result.get('total_cost')}")
            
           
            st.subheader(" Export Results")
            if st.button("Download Results as CSV"):
                csv_buffer = io.StringIO()
                comparison_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label=" Download CSV",
                    data=csv_buffer.getvalue(),
                    file_name="scenario_analysis_results.csv",
                    mime="text/csv"
                )
        else:
            st.info("Run scenario analysis first to see detailed results.")
    
    with tab4:
        st.subheader("🛠 Create Custom Scenarios")
        
       
        st.markdown("### Build Your Own Scenario")
        
        with st.form("custom_scenario_form"):
            scenario_name = st.text_input("Scenario Name", "My Custom Scenario")
            scenario_description = st.text_area("Description", "Custom scenario description")
            
            st.markdown("#### Modifications")
            
           
            labor_enabled = st.checkbox("Modify Labor Availability")
            labor_multiplier = st.slider(
                "Labor Availability Multiplier", 0.1, 3.0, 1.0, 0.1,
                disabled=not labor_enabled
            )
            
           
            material_enabled = st.checkbox("Modify Material Costs")
            if material_enabled and st.session_state.data_processor.resources_df is not None:
                materials = st.session_state.data_processor.resources_df[
                    st.session_state.data_processor.resources_df['resource_type'] == 'material'
                ]['resource_id'].tolist()
                selected_material = st.selectbox("Select Material", materials)
                material_multiplier = st.slider("Cost Multiplier", 0.1, 5.0, 1.0, 0.1)
            
           
            weather_enabled = st.checkbox("Add Weather Delays")
            if weather_enabled and st.session_state.data_processor.tasks_df is not None:
                tasks = st.session_state.data_processor.tasks_df['task_id'].tolist()
                affected_tasks = st.multiselect("Affected Tasks", tasks)
                delay_start = st.number_input("Delay Start Day", 0, 100, 10)
                delay_end = st.number_input("Delay End Day", 0, 100, 15)
            
           
            duration_enabled = st.checkbox("Modify Task Duration")
            if duration_enabled and st.session_state.data_processor.tasks_df is not None:
                task_to_modify = st.selectbox(
                    "Select Task", 
                    st.session_state.data_processor.tasks_df['task_id'].tolist()
                )
                duration_multiplier = st.slider("Duration Multiplier", 0.1, 3.0, 1.0, 0.1)
            
            submitted = st.form_submit_button("Create Scenario")
            
            if submitted and scenario_name:
               
                modifications = []
                
                if labor_enabled:
                    modifications.append({
                        'type': 'change_labor_availability',
                        'multiplier': labor_multiplier
                    })
                
                if material_enabled and 'selected_material' in locals():
                    modifications.append({
                        'type': 'change_material_cost',
                        'material': selected_material,
                        'multiplier': material_multiplier
                    })
                
                if weather_enabled and 'affected_tasks' in locals() and affected_tasks:
                    modifications.append({
                        'type': 'add_weather_delay',
                        'affected_tasks': affected_tasks,
                        'delay_periods': [[delay_start, delay_end]],
                        'weather_type': 'custom_weather'
                    })
                
                if duration_enabled and 'task_to_modify' in locals():
                    modifications.append({
                        'type': 'change_task_duration',
                        'task_id': task_to_modify,
                        'multiplier': duration_multiplier
                    })
                
                if modifications:
                   
                    custom_scenario = Scenario(scenario_name, scenario_description)
                    
                   
                    for mod in modifications:
                        modification_type = mod.pop('type')
                        custom_scenario.add_modification(modification_type, **mod)
                    
                    st.session_state.scenario_analyzer.add_scenario(custom_scenario)
                    st.success(f" Created custom scenario: {scenario_name}")
                else:
                    st.warning("Please enable at least one modification type.")


def simulation_section():
    
   
    if 'baseline_simulation' not in st.session_state:
        st.session_state.baseline_simulation = None
    
   
    tab1, tab2, tab3 = st.tabs(["Quick Simulation", "Real-Time Adjustments", "Impact Tracking"])
    
    with tab1:
        st.subheader("Quick Parameter Testing")
        
       
        if st.session_state.baseline_simulation is None:
            if st.button(" Establish Baseline", type="primary"):
                with st.spinner("Calculating baseline..."):
                    processor = st.session_state.data_processor
                    scheduler_data = processor.preprocess_for_scheduler()
                    scheduler = ConstructionScheduler(100)
                    
                    for task in scheduler_data['tasks'].values():
                        scheduler.add_task(task)
                    for resource in scheduler_data['resources'].values():
                        scheduler.add_resource(resource)
                    for weather_constraint in scheduler_data['weather_constraints']:
                        scheduler.add_weather_constraint(weather_constraint)
                    
                    result = scheduler.solve(time_limit_seconds=30)
                    st.session_state.baseline_simulation = result
                    st.rerun()
        else:
            baseline = st.session_state.baseline_simulation
            
           
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Baseline Makespan", f"{baseline['makespan']} days")
            with col2:
                st.metric("Baseline Cost", f"${baseline['total_cost']:,.2f}")
            with col3:
                st.metric("Status", baseline['status'])
            with col4:
                st.metric("Solve Time", f"{baseline['solver_stats']['solve_time']:.2f}s")
            
            st.markdown("---")
            
           
            st.subheader("🔧 Adjust Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Resource Changes**")
                labor_multiplier = st.slider("Labor Availability", 0.3, 3.0, 1.0, 0.1, 
                                           help="Multiplier for labor capacity")
                material_cost_multiplier = st.slider("Material Cost Factor", 0.5, 3.0, 1.0, 0.1,
                                                    help="Multiplier for material costs")
                
            with col2:
                st.markdown("**Schedule Pressure**")
                duration_pressure = st.slider("Duration Pressure", 0.5, 1.5, 1.0, 0.05,
                                             help="Reduce durations (crash schedule)")
                weather_impact = st.slider("Weather Severity", 0.0, 2.0, 1.0, 0.1,
                                         help="Multiplier for weather delays")
            
           
            if st.button(" Run Quick Simulation"):
                with st.spinner("Simulating scenario..."):
                   
                    temp_analyzer = ScenarioAnalyzer(st.session_state.data_processor)
                    temp_scenario = Scenario("Quick Sim", "Real-time simulation")
                    
                   
                    if labor_multiplier != 1.0:
                        temp_scenario.add_modification('change_labor_availability', multiplier=labor_multiplier)
                    
                    if material_cost_multiplier != 1.0:
                       
                        common_materials = ['steel', 'cement', 'wood']
                        for material in common_materials:
                            temp_scenario.add_modification('change_material_cost', 
                                                         material=material, multiplier=material_cost_multiplier)
                    
                    if duration_pressure != 1.0:
                       
                        for task_id in st.session_state.data_processor.tasks_df['task_id']:
                            temp_scenario.add_modification('change_task_duration', 
                                                         task_id=task_id, multiplier=duration_pressure)
                    
                    temp_analyzer.add_scenario(temp_scenario)
                    
                   
                    sim_results = temp_analyzer.run_scenario_analysis(['Quick Sim'], algorithm='ortools')
                    
                    if 'Quick Sim' in sim_results and sim_results['Quick Sim']['status'] in ['OPTIMAL', 'FEASIBLE']:
                        sim_result = sim_results['Quick Sim']
                        
                       
                        makespan_change = sim_result['makespan'] - baseline['makespan']
                        makespan_pct = (makespan_change / baseline['makespan']) * 100
                        cost_change = sim_result['total_cost'] - baseline['total_cost']
                        cost_pct = (cost_change / baseline['total_cost']) * 100
                        
                       
                        st.subheader(" Simulation Results")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("New Makespan", f"{sim_result['makespan']} days", 
                                    f"{makespan_change:+.0f} days")
                        with col2:
                            st.metric("New Cost", f"${sim_result['total_cost']:,.2f}", 
                                    f"${cost_change:+,.2f}")
                        with col3:
                            st.metric("Schedule Impact", f"{makespan_pct:+.1f}%")
                        with col4:
                            st.metric("Cost Impact", f"{cost_pct:+.1f}%")
                        
                       
                        if makespan_change != 0 or cost_change != 0:
                            st.subheader("Impact Visualization")
                            
                            categories = ['Makespan (days)', 'Cost ($)']
                            baseline_values = [baseline['makespan'], baseline['total_cost']]
                            simulation_values = [sim_result['makespan'], sim_result['total_cost']]
                            
                            fig = go.Figure()
                            fig.add_trace(go.Bar(name='Baseline', x=categories, y=baseline_values, 
                                               marker_color='lightblue'))
                            fig.add_trace(go.Bar(name='Simulation', x=categories, y=simulation_values, 
                                               marker_color='orange'))
                            
                            fig.update_layout(
                                title='Baseline vs Simulation Comparison',
                                barmode='group',
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                    else:
                        st.error("Simulation failed. Try less extreme parameter changes.")
    
    with tab2:
        st.subheader(" Interactive Parameter Adjustment")
        
        if st.session_state.baseline_simulation is not None:
            st.markdown("**Drag sliders to see real-time impact estimates**")
            
           
            col1, col2 = st.columns(2)
            
            with col1:
                auto_labor = st.slider("Labor Capacity", 0.5, 2.0, 1.0, 0.1, key="auto_labor")
                auto_cost = st.slider("Material Costs", 0.7, 2.0, 1.0, 0.1, key="auto_cost")
                
            with col2:
                auto_duration = st.slider("Task Durations", 0.7, 1.3, 1.0, 0.05, key="auto_duration")
                auto_weather = st.slider("Weather Delays", 0.0, 2.0, 1.0, 0.1, key="auto_weather")
            
           
            baseline = st.session_state.baseline_simulation
            
           
            estimated_makespan_impact = 0
            estimated_cost_impact = 0
            
           
            if auto_labor < 1.0:
                estimated_makespan_impact += (1.0 - auto_labor) * 20 
            elif auto_labor > 1.0:
                estimated_makespan_impact -= (auto_labor - 1.0) * 10
            
           
            estimated_makespan_impact += (auto_duration - 1.0) * baseline['makespan']
            
           
            estimated_cost_impact += (auto_cost - 1.0) * baseline['total_cost'] * 0.4 
            
           
            estimated_makespan_impact += (auto_weather - 1.0) * 5 
            
           
            new_makespan = baseline['makespan'] + estimated_makespan_impact
            new_cost = baseline['total_cost'] + estimated_cost_impact
            
            st.subheader(" Estimated Impact")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Estimated Makespan", f"{new_makespan:.0f} days", 
                        f"{estimated_makespan_impact:+.0f} days")
            with col2:
                st.metric("Estimated Cost", f"${new_cost:,.0f}", 
                        f"${estimated_cost_impact:+,.0f}")
            with col3:
                makespan_pct = (estimated_makespan_impact / baseline['makespan']) * 100
                st.metric("Schedule Change", f"{makespan_pct:+.1f}%")
            with col4:
                cost_pct = (estimated_cost_impact / baseline['total_cost']) * 100
                st.metric("Cost Change", f"{cost_pct:+.1f}%")
            
            st.info("💡 These are rough estimates. Use 'Run Quick Simulation' for precise calculations.")
            
        else:
            st.info("Establish baseline first in the Quick Simulation tab.")
    
    with tab3:
        st.subheader("Impact Tracking History")
        
       
        if 'simulation_history' not in st.session_state:
            st.session_state.simulation_history = []
        
        if st.session_state.simulation_history:
            history_df = pd.DataFrame(st.session_state.simulation_history)
            
           
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=history_df.index,
                y=history_df['makespan'],
                mode='lines+markers',
                name='Makespan (days)',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=history_df.index,
                y=history_df['cost'] / 1000, 
                mode='lines+markers',
                name='Cost ($000s)',
                line=dict(color='red'),
                yaxis='y2'
            ))
            
            fig.update_layout(
                title='Simulation History',
                xaxis_title='Simulation Run',
                yaxis_title='Makespan (days)',
                yaxis2=dict(
                    title='Cost ($000s)',
                    overlaying='y',
                    side='right'
                ),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
           
            st.subheader("Simulation Log")
            display_df = history_df.copy()
            display_df['cost'] = display_df['cost'].apply(lambda x: f"${x:,.2f}")
            st.dataframe(display_df, use_container_width=True)
            
            if st.button("Clear History"):
                st.session_state.simulation_history = []
                st.rerun()
                
        else:
            st.info("Run simulations to see tracking history here.")
        
       
        if st.button(" Save Current Parameters to History"):
            if st.session_state.baseline_simulation is not None:
               
                labor = st.session_state.get('auto_labor', 1.0)
                cost_mult = st.session_state.get('auto_cost', 1.0)
                duration = st.session_state.get('auto_duration', 1.0)
                weather = st.session_state.get('auto_weather', 1.0)
                
               
                baseline = st.session_state.baseline_simulation
                makespan_est = baseline['makespan'] * duration
                cost_est = baseline['total_cost'] * cost_mult
                
                entry = {
                    'timestamp': pd.Timestamp.now().strftime('%H:%M:%S'),
                    'labor_mult': labor,
                    'cost_mult': cost_mult,
                    'duration_mult': duration,
                    'weather_mult': weather,
                    'makespan': makespan_est,
                    'cost': cost_est
                }
                
                st.session_state.simulation_history.append(entry)
                st.success("Parameters saved to history!")
                st.rerun()


def main():
    
   
    st.set_page_config(
        page_title="AI Construction Scheduler",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
   
    initialize_session_state()
    
   
    st.title(" AI-Enhanced Construction Scheduling & Resource Allocation")
    st.markdown("""
    Optimize construction schedules and resource allocation using AI-powered algorithms.
    Upload your project data or use sample data to get started.
    """)
    
   
    st.sidebar.title("Navigation")
    
    page = st.sidebar.radio(
        "Choose a section:",
        ["Data Input", "Optimization", "Results", "Scenario Analysis", "Simulation"]
    )
    
   
    with st.sidebar.expander("About This App"):
        st.markdown("""
        This application uses:
        - **OR-Tools CP-SAT** for constraint-based scheduling
        - **Genetic Algorithms** for resource optimization
        - **Interactive visualizations** with Plotly
        - **Comprehensive scenario analysis** for what-if studies
        - **Real-time scenario simulation**
        
        Upload CSV files with your project data or use the sample data to explore the features.
        """)
    
   
    if page == "Data Input":
        data_input_section()
    elif page == "Optimization":
        optimization_section()
    elif page == "Results":
        results_section()
    elif page == "Scenario Analysis":
        scenario_analysis_section()
    elif page == "Simulation":
        simulation_section()
    
   
    st.markdown("---")
    st.markdown(
        "Built with  using Streamlit, OR-Tools, and Plotly | "
        "AI-Enhanced Construction Scheduling System"
    )


if __name__ == "__main__":
    main()