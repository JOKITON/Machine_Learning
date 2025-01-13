""" Program that creates a Multilayer Perceptron model to detect type of cancer cells. """

def init_fb_sig():
    from config import LEARNING_RATE, STEP_SIZE, DECAY_RATE, CONVERGENCE_THRESHOLD
    from config import EPOCHS_FBATCH_3, LS_SIGMOID_1, N_LAYERS, SEED_FB_SIG
    from preprocessing import get_train_test_pd
    from activations import sigmoid, der_sigmoid
    from plot import Plot
    from plots import plot_acc_epochs, plot_loss_epochs
    from setup import setup_layers
    from evaluate import print_preds
    import json
    
    EPOCHS = EPOCHS_FBATCH_3
    LAYER_SHAPE = LS_SIGMOID_1

    # Normalize the data
    X_train, y_train, X_test, y_test = get_train_test_pd()
    y_train = y_train.to_numpy().reshape(-1, 1)
    y_test = y_test.to_numpy().reshape(-1, 1)
    
    with open(SEED_FB_SIG, 'r') as file:
        data = json.load(file)
        seed = int(data['seed'])
        epochs = int(data['epoch'])
        EPOCHS = epochs

    layers = setup_layers(sigmoid, der_sigmoid, LAYER_SHAPE, seed)

    activations = [None] * N_LAYERS

    plot = Plot(X_train, X_test, y_train, y_test, EPOCHS)

    for epoch in range(EPOCHS):
        if epoch % STEP_SIZE == 0:
            LEARNING_RATE *= DECAY_RATE
        acc_train, mse_train, _ = plot.append_preds(layers)

        if (epoch % 100 == 0):
            print(f"Epoch: {epoch}", "MSE: ", f"{mse_train:.5f}", "R2: ", f"{acc_train:.5f}")

        # Forward propagation
        train_input = X_train
        for i in range(N_LAYERS):
            activations[i], _ = layers[i].forward(train_input)
            train_input = activations[i]

        # Backward propagation
        for i in reversed(range(N_LAYERS)):
            layers[i].backward(y_train, LEARNING_RATE)

    plot_acc_epochs(plot.acc_train, plot.acc_test, EPOCHS)
    plot_loss_epochs(plot.mse_train, plot.mse_test, EPOCHS)

    print_preds(layers, X_train, y_train, 1)
    print_preds(layers, X_test, y_test, 2)
    
    return layers
