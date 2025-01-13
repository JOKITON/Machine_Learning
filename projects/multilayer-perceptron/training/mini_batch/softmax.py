""" Program that creates a Multilayer Perceptron model to detect type of cancer cells. """

def init_mb_soft():
    from config import LEARNING_RATE, STEP_SIZE, DECAY_RATE, CONVERGENCE_THRESHOLD
    from config import EPOCHS_MINI_BATCH_3, LS_SOFTMAX_1, N_LAYERS, BATCH_SIZE
    from config import SEED_MB_SOFT
    import numpy as np
    from batch import get_batches, shuffle_batches, get_val_batches
    from preprocessing import get_train_test_pd
    from activations import softmax, der_softmax
    from plot import Plot
    from plots import plot_acc_epochs, plot_loss_epochs
    from setup import setup_layers
    from evaluate import print_preds
    import json

    EPOCHS = EPOCHS_MINI_BATCH_3
    LAYER_SHAPE = LS_SOFTMAX_1

    # Normalize the data
    X_train, y_train, X_test, y_test = get_train_test_pd()
    y_train = y_train.to_numpy().reshape(-1, 1)
    y_test = y_test.to_numpy().reshape(-1, 1)

    with open(SEED_MB_SOFT, 'r') as file:
        data = json.load(file)
        seed = int(data['seed'])
        epochs = int(data['epoch'])
        EPOCHS = epochs
    layers = setup_layers(softmax, der_softmax, LAYER_SHAPE, seed)

    train_x, train_y = get_batches(X_train, y_train, BATCH_SIZE)

    activations = [None] * N_LAYERS

    y_train_softmax = np.zeros((y_train.shape[0], 2))
    y_train_softmax[np.arange(y_train.shape[0]), y_train.flatten()] = 1

    y_test_softmax = np.zeros((y_test.shape[0], 2))
    y_test_softmax[np.arange(y_test.shape[0]), y_test.flatten()] = 1

    plot = Plot(X_train, X_test, y_train_softmax, y_test_softmax, EPOCHS)

    for epoch in range(EPOCHS):
        # Forward propagation
        if epoch % STEP_SIZE == 0:
            LEARNING_RATE *= DECAY_RATE
        acc_train, mse_train, mae_train = plot.append_preds(layers)
        
        if (epoch % 100 == 0):
            print(f"Epoch: {epoch}", "MSE: ", f"{mse_train:.5f}", "R2: ", f"{acc_train:.5f}")
        
        if (epoch % 15 == 0 and epoch != 0):
            train_x, train_y = shuffle_batches(train_x, train_y)

        batch_X, batch_Y = get_val_batches(train_x, train_y, layers, epoch)
        for i in range(N_LAYERS):
            activations[i], output = layers[i].forward(batch_X)
            batch_X = activations[i]
        # print(activations[-1])

        for i in reversed(range(N_LAYERS)):
            if (i == N_LAYERS - 1):
                y_true_one_hot = np.zeros((batch_Y.shape[0], 2))
                y_true_one_hot[np.arange(batch_Y.shape[0]), batch_Y.flatten()] = 1
                input_y = y_true_one_hot
            else:
                input_y = batch_Y
            layers[i].backward(input_y, LEARNING_RATE)

    plot_acc_epochs(plot.acc_train, plot.acc_test, EPOCHS)
    plot_loss_epochs(plot.mse_train, plot.mse_test, EPOCHS)


    print_preds(layers, X_train, y_train_softmax, 1)
    print_preds(layers, X_test, y_test_softmax, 2)
