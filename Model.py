from tensorflow import keras as keras
from keras import layers, regularizers
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam,SGD
#from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from Baseline import Grid_search
from CMA import Adam_cma
from Random import global_random_search
from Annealing import simulated_annealing

def Adam_get_model_loss(alpha, beta1, beta2):
    batch_size = 128
    epochs = 10
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape((60000, 28, 28, 1))
    x_test = x_test.reshape((10000, 28, 28, 1))
    n = 5000
    x_train = x_train[1:n]; y_train=y_train[1:n]
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    optimizer = Adam(learning_rate=alpha, beta_1=beta1, beta_2=beta2)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer,
        metrics=["accuracy"])
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
        validation_split=0.1,verbose = 2)
    model.save("cifar.model")
    y_preds = model.predict(x_test)
    loss = CategoricalCrossentropy()
    return loss(y_test, y_preds).numpy()

def Adam_paras():
    bounds = [
            [0.001, 0.011],
            [0.25, 0.91],
            [0.900, 1]
        ]
    initial_solution = [0.001, 0.25, 0.9]
    initial_temperature = 10
    cooling_rate = 0.94
    max_iterations = 200
    step_size = [0.001, 0.65, 0.01]
    Grid_xs,Grid_fs = Grid_search(Adam_get_model_loss,bounds,0.001,0.65,0.01)
    Cma_xs,Cma_fs = Adam_cma(Adam_get_model_loss)
    random_xs,random_fs = global_random_search(Adam_get_model_loss,bounds,30)
    Ann_xs,Ann_fs = simulated_annealing(Adam_get_model_loss,initial_solution, initial_temperature, cooling_rate, max_iterations, step_size)

    plot_results(Grid_fs,Cma_fs,random_fs,Ann_fs)


def plot_results(Grid_fs,Cma_fs,random_fs,Ann_fs):
    Grid_N = len(Grid_fs)
    Grid_X = list(range(Grid_N))
    Cma_N = len(Cma_fs)
    Cma_X = list(range(Cma_N))
    Random_N = len(random_fs)
    Random_X = list(range(Random_N))
    Ann_N = len(Ann_fs)
    Ann_X = list(range(Ann_N))
    plt.plot(Grid_X, Grid_fs, label='Base Line')
    plt.plot(Cma_X, Cma_fs, label='CMA-ES')
    plt.plot(Random_X,random_fs,label='Random search')
    plt.plot(Ann_X,Ann_fs,label='Annealing')
    plt.xlabel('Optimisation Iterations')
    plt.ylabel('Model Loss')
    plt.ylim(0,0.5)
    plt.legend()
    plt.show()

Adam_paras()

# initial_solution = [0.001,0.25,0.9]
# initial_temperature = 10
# cooling_rate = 0.94
# max_iterations = 200
# step_size = [0.001,0.65,0.01]
# solution, energy = simulated_annealing(Adam_get_model_loss,initial_solution, initial_temperature, cooling_rate, max_iterations, step_size)
# print("Solution: ", solution)
# print("Energy: ", energy)