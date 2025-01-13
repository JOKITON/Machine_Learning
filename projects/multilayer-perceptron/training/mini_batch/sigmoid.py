""" Program that creates a Multilayer Perceptron model to detect type of cancer cells. """

def init_mb_sig():
    import config
    from config import LEARNING_RATE, STEP_SIZE, DECAY_RATE, CONVERGENCE_THRESHOLD
    from config import EPOCHS_MINI_BATCH_2, LS_SIGMOID_2, N_LAYERS, BATCH_SIZE, SEED_MB_SIG
    from colorama import Fore, Back, Style
    from preprocessing import get_train_test_pd
    from activations import sigmoid, relu, der_sigmoid, der_relu, leaky_relu, der_leaky_relu, tanh, der_tanh, softmax, der_softmax
    from plot import Plot
    from plots import plot_acc_epochs, plot_loss_epochs
    from setup import setup_layers
    from evaluate import make_preds
    from evaluate import print_preds
    import json

    RESET_ALL = Fore.RESET + Back.RESET + Style.RESET_ALL
    EPOCHS = EPOCHS_MINI_BATCH_2
    LAYER_SHAPE = LS_SIGMOID_2

    # Normalize the data
    X_train, y_train, X_test, y_test = get_train_test_pd()
    y_train = y_train.to_numpy().reshape(-1, 1)
    y_test = y_test.to_numpy().reshape(-1, 1)
    
    with open(SEED_MB_SIG, 'r') as file:
        data = json.load(file)
        seed = int(data['seed'])
        epochs = int(data['epoch'])

    layers = setup_layers(sigmoid, der_sigmoid, LAYER_SHAPE, seed)

    activations = [None] * N_LAYERS

    plot = Plot(X_train, X_test, y_train, y_test, epochs)

    for epoch in range(epochs):
        # Forward propagation
        if epoch % STEP_SIZE == 0:
            LEARNING_RATE *= DECAY_RATE
        acc_train, mse_train, mae_train = plot.append_preds(layers)
        
        if (epoch % 100 == 0):
            print(f"Epoch: {epoch}", "MSE: ", f"{mse_train:.5f}", "R2: ", f"{acc_train:.5f}")

        train_input = X_train
        for i in range(N_LAYERS):
            activations[i], output = layers[i].forward(train_input)
            train_input = activations[i]

        for i in reversed(range(N_LAYERS)):
            layers[i].backward(y_train, LEARNING_RATE)

    plot_acc_epochs(plot.acc_train, plot.acc_test, epochs)
    plot_loss_epochs(plot.mse_train, plot.mse_test, epochs)

    print_preds(layers, X_train, y_train, 1)
    print_preds(layers, X_test, y_test, 2)