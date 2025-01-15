""" Program that creates a Multilayer Perceptron model to detect type of cancer cells. """

def init_fb_soft(plt=None, plt_it=0, plt_ct=2, plt_show=True, plt_ret=False):
    from config import LEARNING_RATE, STEP_SIZE, DECAY_RATE, CONVERGENCE_THRESHOLD, SEED_FB_SOFT
    from config import EPOCHS_FBATCH_2, LS_SOFTMAX_0, N_LAYERS
    import numpy as np
    from preprocessing import get_train_test_pd
    from activations import softmax, der_softmax
    from plot import Plot
    from setup import setup_layers
    from evaluate import print_preds
    import json

    EPOCHS = EPOCHS_FBATCH_2
    LAYER_SHAPE = LS_SOFTMAX_0
    LEARNING_RATE *= 7.5
    COUNT_PLOT = plt_ct

    # Normalize the data
    X_train, y_train, X_test, y_test = get_train_test_pd()
    y_train = y_train.to_numpy().reshape(-1, 1)
    y_test = y_test.to_numpy().reshape(-1, 1)
    
    with open(SEED_FB_SOFT, 'r') as file:
        data = json.load(file)
        seed = int(data['seed'])
        epochs = int(data['epoch'])
        EPOCHS = epochs

    layers = setup_layers(softmax, der_softmax, LAYER_SHAPE, seed)

    activations = [None] * N_LAYERS

    soft_y_train = np.zeros((y_train.shape[0], 2))
    soft_y_train[np.arange(y_train.shape[0]), y_train.flatten()] = 1

    soft_y_test = np.zeros((y_test.shape[0], 2))
    soft_y_test[np.arange(y_test.shape[0]), y_test.flatten()] = 1

    if (plt == None):
        plt = Plot(COUNT_PLOT)
    plt.set_error_data(X_train, soft_y_train, layers, plt_it)
    plt.set_plot_config("Training", "blue", plt_it, "-", EPOCHS)
    plt.set_error_data(X_test, soft_y_test, layers, plt_it + 1)
    plt.set_plot_config("Validation", "orange", plt_it + 1, "--", EPOCHS)

    for epoch in range(EPOCHS):
        if epoch % STEP_SIZE == 0:
            LEARNING_RATE *= DECAY_RATE

        plt.set_error_data(X_train, soft_y_train, layers, plt_it)
        plt.set_error_data(X_test, soft_y_test, layers, plt_it + 1)
        acc_train, mse_train, _ = plt.get_error_data(plt_it)

        if (epoch % 100 == 0):
            print(f"Epoch: {epoch}", "MSE: ", f"{mse_train:.5f}", "R2: ", f"{acc_train:.5f}")

        # Forward propagation
        train_input = X_train
        for i in range(N_LAYERS):
            activations[i], _ = layers[i].forward(train_input)
            train_input = activations[i]

        # Backward propagation
        for i in reversed(range(N_LAYERS)):
            if (i == N_LAYERS - 1):
                soft_y_true = np.zeros((y_train.shape[0], 2))
                soft_y_true[np.arange(y_train.shape[0]), y_train.flatten()] = 1
                input_y = soft_y_true
            else:
                input_y = y_train
            layers[i].backward(input_y, LEARNING_RATE)

    if (plt_show):
        plt.plot_acc_epochs()
        plt.plot_loss_epochs()

    print_preds(layers, X_train, soft_y_train, 1)
    print_preds(layers, X_test, soft_y_test, 2)
    
    if (plt_ret):
        return layers, plt
    else:
        return layers
