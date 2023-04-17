import numpy as np
def Grid_search(f,bound,step_size_alpha,step_size_beta1,step_size_beta2):
    min_value = float("inf")
    best_para = [0,0,0]
    x_history = []
    f_history = []
    for x in np.arange(bound[0][0], bound[0][1],step_size_alpha):
        for y in np.arange(bound[1][0], bound[1][1],step_size_beta1):
            for z in np.arange(bound[2][0], bound[2][1],step_size_beta2):
                current_value = f(x,y,z)
                if current_value < min_value:
                    min_value = current_value
                    best_para = [x,y,z]
                x_history.append([x,y,z])
                f_history.append(min_value)
    print(min_value)
    print(best_para)
    return x_history,f_history
