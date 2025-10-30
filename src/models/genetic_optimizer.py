"""
Genetic Algorithm for Dynamic Resource Allocation in Construction Projects.

This module implements genetic algorithms to optimize resource allocation
across multiple construction tasks, considering constraints and objectives.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import random
from dataclasses import dataclass
from copy import deepcopy


@dataclass
class ResourceAllocation:
    """Represents resource allocation for a specific task."""
    task_id: str
    resource_assignments: Dict[str, int]  # resource_id -> amount allocated
    start_time: int
    priority: float = 1.0


class Individual:
    """Represents an individual solution in the genetic algorithm."""
    
    def __init__(self, allocations: List[ResourceAllocation]):
        self.allocations = allocations
        self.fitness = 0.0
        self.is_valid = True
        
    def __str__(self):
        return f"Individual(fitness={self.fitness:.3f}, valid={self.is_valid})"


class ResourceOptimizer:
    """Genetic Algorithm optimizer for construction resource allocation."""
    
    def __init__(self, 
                 population_size: int = 100,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 elitism_rate: float = 0.1,
                 max_generations: int = 500):
        
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        self.max_generations = max_generations
        
        self.tasks = {}
        self.resources = {}
        self.constraints = []
        self.population = []
        self.best_individual = None
        self.fitness_history = []
        
    def add_task(self, task_id: str, resource_requirements: Dict[str, Tuple[int, int]], 
                 duration: int, priority: float = 1.0, deadline: Optional[int] = None):
        """Add a task with resource requirements (min, max) for each resource type."""
        self.tasks[task_id] = {
            'resource_requirements': resource_requirements,
            'duration': duration,
            'priority': priority,
            'deadline': deadline
        }
        
    def add_resource(self, resource_id: str, total_capacity: int, cost_per_unit: float = 1.0):
        """Add a resource with total capacity and cost."""
        self.resources[resource_id] = {
            'total_capacity': total_capacity,
            'cost_per_unit': cost_per_unit,
            'allocated': 0
        }
        
    def add_constraint(self, constraint_type: str, **kwargs):
        """Add various types of constraints."""
        self.constraints.append({
            'type': constraint_type,
            'params': kwargs
        })
        
    def create_random_individual(self) -> Individual:
        """Create a random individual solution."""
        allocations = []
        
        for task_id, task_info in self.tasks.items():
            resource_assignments = {}
            
            for resource_id, (min_req, max_req) in task_info['resource_requirements'].items():
                if resource_id in self.resources:
                    allocated_amount = random.randint(min_req, max_req)
                    resource_assignments[resource_id] = allocated_amount
                    
            start_time = random.randint(0, 50)
            
            allocation = ResourceAllocation(
                task_id=task_id,
                resource_assignments=resource_assignments,
                start_time=start_time,
                priority=task_info['priority']
            )
            allocations.append(allocation)
            
        return Individual(allocations)
    
    def initialize_population(self):
        """Initialize the population with random individuals."""
        self.population = []
        for _ in range(self.population_size):
            individual = self.create_random_individual()
            self.population.append(individual)
            
    def calculate_fitness(self, individual: Individual) -> float:
        """Calculate fitness score for an individual."""
        total_fitness = 0.0
        resource_usage = {rid: 0 for rid in self.resources.keys()}
        
        for allocation in individual.allocations:
            for resource_id, amount in allocation.resource_assignments.items():
                resource_usage[resource_id] += amount
                
        capacity_penalty = 0
        for resource_id, used in resource_usage.items():
            if used > self.resources[resource_id]['total_capacity']:
                capacity_penalty += (used - self.resources[resource_id]['total_capacity']) * 10
                individual.is_valid = False
                
        utilization_score = 0
        for resource_id, used in resource_usage.items():
            capacity = self.resources[resource_id]['total_capacity']
            if capacity > 0:
                utilization = min(used / capacity, 1.0)
                utilization_score += utilization
                
        total_cost = 0
        for allocation in individual.allocations:
            for resource_id, amount in allocation.resource_assignments.items():
                cost_per_unit = self.resources[resource_id]['cost_per_unit']
                total_cost += amount * cost_per_unit
                
        makespan = 0
        for allocation in individual.allocations:
            task_id = allocation.task_id
            task_duration = self.tasks[task_id]['duration']
            completion_time = allocation.start_time + task_duration
            makespan = max(makespan, completion_time)
            
        priority_score = 0
        for allocation in individual.allocations:
            task_priority = allocation.priority
            completion_time = allocation.start_time + self.tasks[allocation.task_id]['duration']
            priority_score += task_priority / max(completion_time, 1)
            
        deadline_penalty = 0
        for allocation in individual.allocations:
            task_deadline = self.tasks[allocation.task_id].get('deadline')
            if task_deadline:
                completion_time = allocation.start_time + self.tasks[allocation.task_id]['duration']
                if completion_time > task_deadline:
                    deadline_penalty += (completion_time - task_deadline) * 5
                    
        fitness = (
            utilization_score * 100 +           # Resource utilization
            priority_score * 50 +               # Priority-weighted efficiency  
            max(0, 100 - makespan) +            # Makespan minimization
            max(0, 100 - total_cost/10) -       # Cost minimization
            capacity_penalty -                  # Capacity constraint penalty
            deadline_penalty                    # Deadline penalty
        )
        
        return max(fitness, 0.1)  # Ensure positive fitness
    
    def selection(self, tournament_size: int = 3) -> Individual:
        """Tournament selection."""
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Single-point crossover between two parents."""
        if random.random() > self.crossover_rate:
            return deepcopy(parent1), deepcopy(parent2)
            
        crossover_point = random.randint(1, len(parent1.allocations) - 1)
        
        child1_allocations = (parent1.allocations[:crossover_point] + 
                            parent2.allocations[crossover_point:])
        child2_allocations = (parent2.allocations[:crossover_point] + 
                            parent1.allocations[crossover_point:])
        
        child1 = Individual(child1_allocations)
        child2 = Individual(child2_allocations)
        
        return child1, child2
    
    def mutate(self, individual: Individual):
        """Mutate an individual by randomly adjusting resource allocations."""
        for allocation in individual.allocations:
            if random.random() < self.mutation_rate:
                for resource_id in allocation.resource_assignments:
                    if resource_id in self.tasks[allocation.task_id]['resource_requirements']:
                        min_req, max_req = self.tasks[allocation.task_id]['resource_requirements'][resource_id]
                        current = allocation.resource_assignments[resource_id]
                        adjustment = random.randint(-2, 2)
                        new_value = max(min_req, min(max_req, current + adjustment))
                        allocation.resource_assignments[resource_id] = new_value
                        
                if random.random() < 0.5:
                    adjustment = random.randint(-5, 5)
                    allocation.start_time = max(0, allocation.start_time + adjustment)
    
    def evolve_generation(self):
        """Evolve one generation."""
        for individual in self.population:
            individual.fitness = self.calculate_fitness(individual)
            
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        if not self.best_individual or self.population[0].fitness > self.best_individual.fitness:
            self.best_individual = deepcopy(self.population[0])
            
        avg_fitness = sum(ind.fitness for ind in self.population) / len(self.population)
        self.fitness_history.append({
            'best_fitness': self.population[0].fitness,
            'avg_fitness': avg_fitness,
            'valid_solutions': sum(1 for ind in self.population if ind.is_valid)
        })
        
        new_population = []
        
        elite_size = int(self.population_size * self.elitism_rate)
        new_population.extend(deepcopy(ind) for ind in self.population[:elite_size])
        
        while len(new_population) < self.population_size:
            parent1 = self.selection()
            parent2 = self.selection()
            
            child1, child2 = self.crossover(parent1, parent2)
            
            self.mutate(child1)
            self.mutate(child2)
            
            new_population.extend([child1, child2])
            
        self.population = new_population[:self.population_size]
    
    def optimize(self, verbose: bool = True) -> Dict:
        """Run the genetic algorithm optimization."""
        self.initialize_population()
        
        for generation in range(self.max_generations):
            self.evolve_generation()
            
            if verbose and generation % 50 == 0:
                best_fitness = self.fitness_history[-1]['best_fitness']
                avg_fitness = self.fitness_history[-1]['avg_fitness']
                valid_count = self.fitness_history[-1]['valid_solutions']
                print(f"Generation {generation}: Best={best_fitness:.2f}, "
                      f"Avg={avg_fitness:.2f}, Valid={valid_count}/{self.population_size}")
                      
        final_fitness = self.calculate_fitness(self.best_individual)
        
        best_allocation = {}
        total_cost = 0
        resource_usage = {rid: 0 for rid in self.resources.keys()}
        
        for allocation in self.best_individual.allocations:
            best_allocation[allocation.task_id] = {
                'resource_assignments': allocation.resource_assignments,
                'start_time': allocation.start_time,
                'end_time': allocation.start_time + self.tasks[allocation.task_id]['duration'],
                'priority': allocation.priority
            }
            
            for resource_id, amount in allocation.resource_assignments.items():
                resource_usage[resource_id] += amount
                total_cost += amount * self.resources[resource_id]['cost_per_unit']
        
        makespan = max(alloc['end_time'] for alloc in best_allocation.values())
        
        return {
            'status': 'OPTIMAL' if self.best_individual.is_valid else 'FEASIBLE',
            'allocation': best_allocation,
            'total_cost': total_cost,
            'makespan': makespan,
            'resource_usage': resource_usage,
            'fitness': final_fitness,
            'generations': len(self.fitness_history),
            'fitness_history': self.fitness_history
        }


def create_sample_resource_problem() -> ResourceOptimizer:
    """Create a sample resource allocation problem for testing."""
    optimizer = ResourceOptimizer(population_size=50, max_generations=200)
    
    optimizer.add_resource('workers', 20, 100)  # 20 workers, $100/day each
    optimizer.add_resource('equipment', 5, 500)  # 5 equipment units, $500/day each
    optimizer.add_resource('materials', 1000, 1)  # 1000 units materials, $1 each
    
    optimizer.add_task('foundation', 
                      {'workers': (3, 8), 'equipment': (1, 2), 'materials': (50, 100)}, 
                      duration=7, priority=1.0, deadline=15)
    
    optimizer.add_task('framing', 
                      {'workers': (4, 10), 'equipment': (0, 1), 'materials': (80, 150)}, 
                      duration=10, priority=0.8, deadline=30)
    
    optimizer.add_task('roofing', 
                      {'workers': (2, 6), 'equipment': (1, 2), 'materials': (30, 60)}, 
                      duration=5, priority=0.9, deadline=40)
    
    optimizer.add_task('electrical', 
                      {'workers': (1, 4), 'equipment': (0, 1), 'materials': (20, 40)}, 
                      duration=8, priority=0.7, deadline=45)
    
    optimizer.add_task('finishing', 
                      {'workers': (3, 8), 'equipment': (0, 1), 'materials': (40, 80)}, 
                      duration=12, priority=0.6, deadline=60)
    
    return optimizer


if __name__ == "__main__":
    optimizer = create_sample_resource_problem()
    result = optimizer.optimize()
    
    print(f"\nOptimization Results:")
    print(f"Status: {result['status']}")
    print(f"Total Cost: ${result['total_cost']:.2f}")
    print(f"Project Makespan: {result['makespan']} days")
    print(f"Final Fitness: {result['fitness']:.2f}")
    
    print(f"\nResource Usage:")
    for resource_id, usage in result['resource_usage'].items():
        capacity = optimizer.resources[resource_id]['total_capacity']
        print(f"{resource_id}: {usage}/{capacity} ({usage/capacity*100:.1f}%)")
    
    print(f"\nTask Allocation:")
    for task_id, allocation in result['allocation'].items():
        print(f"{task_id}: Start day {allocation['start_time']}, "
              f"End day {allocation['end_time']}")
        for resource_id, amount in allocation['resource_assignments'].items():
            print(f"  {resource_id}: {amount}")