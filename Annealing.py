import math
import random

# def get_random_neighbor(x, step_size):
#     alpha = x[0]
#     beta1 = x[1]
#     beta2 = x[2]
#     new_x = (alpha+random.uniform(-step_size[0], step_size[0]),beta1+random.uniform(-step_size[1], step_size[1]),beta2+random.uniform(-step_size[2], step_size[2]))
#     return new_x

def get_random_neighbor(current_solution):
    x_range = [[0.001, 0.1], [0.25, 0.9], [0.9, 0.999]]
    x_i = random.uniform(-0.5, 0.5)
    x_ii = random.uniform(0.8, 1.2)
    x_iii = random.uniform(0.995, 1.005)
    change_i = current_solution[0] * math.pow(10, x_i)
    if change_i > 0.1:
        current_solution[0] = x_range[0][1]
    elif change_i < 0.001:
        current_solution[0] = x_range[0][0]
    else:
        current_solution[0] = change_i
    change_ii = current_solution[1] * x_ii
    if change_ii > 0.9:
        current_solution[1] = x_range[1][1]
    elif change_ii < 0.25:
        current_solution[1] = x_range[1][0]
    else:
        current_solution[1] = change_ii
    change_iii = current_solution[2] * x_iii
    if change_iii > 0.999:
        current_solution[2] = x_range[2][1]
    elif change_iii < 0.9:
        current_solution[2] = x_range[2][0]
    else:
        current_solution[2] = change_iii
    return current_solution

def schedule_temperature(time_step, initial_temperature, cooling_rate):
    return initial_temperature * (cooling_rate ** time_step)

def simulated_annealing(f,initial_solution, initial_temperature, cooling_rate, max_iterations, step_size):
    fs = []
    xs=[]
    current_solution = initial_solution
    current_loss= f(current_solution[0],current_solution[1],current_solution[2])

    for time_step in range(max_iterations):
        new_solution = get_random_neighbor(current_solution)
        new_loss = f(new_solution[0],new_solution[1],new_solution[2])

        temperature = schedule_temperature(time_step, initial_temperature, cooling_rate)

        if new_loss < current_loss:
            current_solution = new_solution
            current_loss = new_loss
        else:
            acceptance_probability = math.exp(-(new_loss - current_loss) / temperature)
            if random.random() < acceptance_probability:
                current_solution = new_solution
                current_loss = new_loss
        fs.append(current_loss)
        xs.append(current_solution)

    return xs,fs



