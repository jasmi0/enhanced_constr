#!/usr/bin/env python3
"""
Test script to verify the AI-Enhanced Construction Scheduling system is working correctly.
Run this script to perform basic functionality tests.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_or_tools_scheduler():
    """Test the OR-Tools scheduler."""
    print("Testing OR-Tools Scheduler...")
    try:
        from src.models.scheduler import create_sample_problem
        
        scheduler = create_sample_problem()
        result = scheduler.solve(time_limit_seconds=30, objective_type='minimize_makespan')
        
        if result['status'] in ['OPTIMAL', 'FEASIBLE']:
            print(f" OR-Tools test passed! Status: {result['status']}, Makespan: {result['makespan']} days")
            return True
        else:
            print(f" OR-Tools test failed! Status: {result['status']}")
            return False
    except Exception as e:
        print(f" OR-Tools test failed with error: {e}")
        return False


def test_genetic_algorithm():
    """Test the genetic algorithm optimizer."""
    print("Testing Genetic Algorithm...")
    try:
        from src.models.genetic_optimizer import create_sample_resource_problem
        
        optimizer = create_sample_resource_problem()
        result = optimizer.optimize(verbose=False)
        
        if result['status'] in ['OPTIMAL', 'FEASIBLE']:
            print(f" Genetic Algorithm test passed! Status: {result['status']}, "
                  f"Cost: ${result['total_cost']:.2f}, Makespan: {result['makespan']} days")
            return True
        else:
            print(f" Genetic Algorithm test failed! Status: {result['status']}")
            return False
    except Exception as e:
        print(f" Genetic Algorithm test failed with error: {e}")
        return False


def test_data_processor():
    """Test the data processing utilities."""
    print("Testing Data Processor...")
    try:
        from src.utils.data_processor import DataProcessor, create_sample_data
        
        processor = DataProcessor()
        tasks_df, resources_df, weather_df = create_sample_data()
        dependencies = [('T1', 'T2'), ('T2', 'T3'), ('T3', 'T4'), ('T3', 'T5')]
        
        success = processor.load_from_dataframes(tasks_df, resources_df, weather_df, dependencies)
        
        if success:
            features = processor.extract_features()
            scheduler_data = processor.preprocess_for_scheduler()
            ga_data = processor.preprocess_for_genetic_algorithm()
            
            print(f" Data Processor test passed! "
                  f"Features: {len(features)}, Tasks: {len(scheduler_data['tasks'])}, "
                  f"Resources: {len(scheduler_data['resources'])}")
            return True
        else:
            print(" Data Processor test failed! Could not load data")
            return False
    except Exception as e:
        print(f" Data Processor test failed with error: {e}")
        return False


def test_visualization():
    """Test the visualization components."""
    print("Testing Visualization...")
    try:
        from src.visualization.charts import create_sample_visualization
        
        gantt_fig, resource_fig, pie_fig = create_sample_visualization()
        
        # Check if figures are created (basic validation)
        if gantt_fig and resource_fig and pie_fig:
            print(" Visualization test passed! All charts created successfully")
            return True
        else:
            print(" Visualization test failed! Could not create charts")
            return False
    except Exception as e:
        print(f" Visualization test failed with error: {e}")
        return False


def test_sample_data():
    """Test loading sample CSV files."""
    print("Testing Sample Data Files...")
    try:
        import pandas as pd
        
        # Test loading all sample files
        tasks_df = pd.read_csv('data/sample/sample_tasks.csv')
        resources_df = pd.read_csv('data/sample/sample_resources.csv')
        weather_df = pd.read_csv('data/sample/sample_weather.csv')
        deps_df = pd.read_csv('data/sample/sample_dependencies.csv')
        
        print(f" Sample data test passed! "
              f"Tasks: {len(tasks_df)}, Resources: {len(resources_df)}, "
              f"Weather: {len(weather_df)}, Dependencies: {len(deps_df)}")
        return True
    except Exception as e:
        print(f" Sample data test failed with error: {e}")
        return False


def main():
    """Run all tests."""
    print("🏗️ AI-Enhanced Construction Scheduling System - Test Suite")
    print("=" * 60)
    
    tests = [
        test_sample_data,
        test_data_processor,
        test_or_tools_scheduler,
        test_genetic_algorithm,
        test_visualization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("To start the application, run: streamlit run app.py")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        
    return passed == total


if __name__ == "__main__":
    main()