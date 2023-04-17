import numpy as np
from copy import deepcopy
def global_random_search(f,bounds,N,istime = False):
    best_x = 0
    best_f = 1e15
    x_history = []
    f_history = []
    for _ in range(N):
        new_x = [np.random.uniform(bounds[j][0], bounds[j][1]) for j in range(len(bounds))]
        new_f = f(*new_x)
        if new_f < best_f:
            best_x = deepcopy(new_x)
            best_f = new_f
        if not istime:
            x_history.append(deepcopy(best_x))
            f_history.append(best_f)
    return x_history,f_history