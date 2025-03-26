from .optimization_planner_their_cost_function import OptimizationPlanner
from .optimization_planner_time import OptimizationPlannerTime
from .configuration_space import BicycleConfigurationSpace, Plan, expanded_obstacles
import time
import numpy as np 

def main():
    start = np.array([1, 1, 0, 0]) 
    goal = np.array([9, 9, 0, 0])
    xy_low = [0, 0]
    xy_high = [10, 10]
    phi_max = 0.6
    u1_max = 2
    u2_max = 3
    obstacles = [[6, 3.5, 1.5], [3.5, 6.5, 1]]

    config = BicycleConfigurationSpace(
        xy_low + [-np.inf, -phi_max],
        xy_high + [np.inf, phi_max],
        [-u1_max, -u2_max],
        [u1_max, u2_max],
        obstacles,
        0.15
    )

    N_val_array = np.array([400])
    
    start_time_stamp = time.perf_counter()
    class_cost_function = OptimizationPlanner(config)
    initialization_theirs_time_stamp = time.perf_counter()
    time_cost_function = OptimizationPlannerTime(config)
    initialization_ours_time_stamp = time.perf_counter()

    init_times = {
        "Time-based": initialization_ours_time_stamp - initialization_theirs_time_stamp,
        "Class-based": initialization_theirs_time_stamp - start_time_stamp
    }
    
    solve_times = []
    
    for N_val in N_val_array:
        start_class = time.perf_counter()
        class_cost_function.reid_big_function(start, goal, N=N_val, dt=0.01)
        end_class = time.perf_counter()
        
        start_time_based = time.perf_counter()
        time_cost_function.reid_big_function(start, goal, N=N_val)
        end_time_based = time.perf_counter()
        
        solve_times.append((N_val, end_class - start_class, end_time_based - start_time_based))
    
    print("Initialization Times:")
    for key, value in init_times.items():
        print(f"{key}: {value:.4f} seconds")
    
    print("\nSolve Times:")
    for N_val, class_time, time_based_time in solve_times:
        print(f"N={N_val}: Class-based = {class_time:.4f} seconds, Time-based = {time_based_time:.4f} seconds")

if __name__ == '__main__':
    main()
