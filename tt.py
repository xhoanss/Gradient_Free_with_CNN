x_range = [[0.001, 0.1], [0.25, 0.9], [0.9, 0.999]]
    bounds = np.array(x_range)
    initial = []
    solutions = []
    for _ in range(7):
        a = [random.uniform(x_range[0][0], x_range[0][1]), random.uniform(x_range[1][0], x_range[1][1]), random.uniform(x_range[2][0], x_range[2][1])]
        a = np.array(a)
        initial.append(a)
    optimizer = CMA(mean=np.array([0.001, 0.25, 0.9]), sigma=0.1, bounds=bounds)
    for x in initial:
        loss, acc = average_loss(128, x[0], x[1], x[2], 20)
        solutions.append((x, loss))
        print(f"#{0} {loss} (a={x[0]}, b1 = {x[1]}, b2 = {x[2]})")
    optimizer.tell(solutions)

    for generation in range(50):
        solutions = []
        x_list = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            x_list.append(x)
        for x in x_list:
            loss, acc = average_loss(128, x[0], x[1], x[2], 20)
            solutions.append((x, loss))
            print(f"#{generation} {loss} (a={x[0]}, b1 = {x[1]}, b2 = {x[2]})")
        optimizer.tell(solutions)