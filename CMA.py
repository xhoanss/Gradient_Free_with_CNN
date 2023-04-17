import numpy as np
from cmaes import CMA
#随机生成6个点，计算value，solution append，然后tell, 省去了自己找的过程
def Adam_cma(f):
    xs = []
    fs = []
    op_range = [[0.001, 0.1], [0.25, 0.9], [0.9, 0.99]]
    bounds = np.array(op_range)
    initial = []
    solutions = []
    for _ in range(7):
        x = [np.random.uniform(op_range[0][0], op_range[0][1]), np.random.uniform(op_range[1][0], op_range[1][1]),
             np.random.uniform(op_range[2][0], op_range[2][1])]
        x = np.array(x)
        initial.append(x)
    optimizer = CMA(mean=np.array([0.001, 0.25, 0.9]), sigma=0.1, bounds=bounds)
    for x in initial:
        value = f(x[0], x[1],x[2])
        solutions.append((x, value))
    optimizer.tell(solutions)

    for generation in range(50):
        solutions = []
        max = 0
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = f(x[0], x[1],x[2])
            solutions.append((x, value))
            if value>max:
                max = value
                fs.append(value)
                xs.append(x)
            print(f"#{generation} {value} (x1={x[0]}, x2 = {x[1]}),x3 = {x[2]}")
        optimizer.tell(solutions)

    return xs,fs


