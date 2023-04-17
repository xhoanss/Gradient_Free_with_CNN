import math
import random

# 定义一个优化问题的目标函数，例如求解函数f(x) = x^2的最小值
def objective_function(x):
    return x ** 2

# 随机生成新解
def get_random_neighbor(x, step_size):
    return x + random.uniform(-step_size, step_size)

# 定义一个温度调度函数
def schedule_temperature(time_step, initial_temperature, cooling_rate):
    return initial_temperature * (cooling_rate ** time_step)

# 模拟退火算法
def simulated_annealing(initial_solution, initial_temperature, cooling_rate, max_iterations, step_size):
    current_solution = initial_solution
    current_energy = objective_function(current_solution)

    for time_step in range(max_iterations):
        new_solution = get_random_neighbor(current_solution, step_size)
        new_energy = objective_function(new_solution)

        temperature = schedule_temperature(time_step, initial_temperature, cooling_rate)

        if new_energy < current_energy:
            current_solution = new_solution
            current_energy = new_energy
        else:
            acceptance_probability = math.exp(-(new_energy - current_energy) / temperature)
            if random.random() < acceptance_probability:
                current_solution = new_solution
                current_energy = new_energy

    return current_solution, current_energy

# 参数设置
initial_solution = random.uniform(-10, 10)
initial_temperature = 100
cooling_rate = 0.99
max_iterations = 1000
step_size = 0.1

# 求解
solution, energy = simulated_annealing(initial_solution, initial_temperature, cooling_rate, max_iterations, step_size)
print("Solution: ", solution)
print("Energy: ", energy)
