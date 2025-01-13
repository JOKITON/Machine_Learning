""" Program that creates a Multilayer Perceptron model to detect type of cancer cells. """

def init_fb_sig_seed():
    from config import LEARNING_RATE, STEP_SIZE, DECAY_RATE, CONVERGENCE_THRESHOLD
    from config import EPOCHS_FBATCH_3, LS_SIGMOID_2, N_LAYERS, BATCH_SIZE
    from preprocessing import get_train_test_pd
    from activations import sigmoid, der_sigmoid
    from loss import f_r2score
    import numpy as np
    from setup import setup_layers

    EPOCHS = EPOCHS_FBATCH_3
    LAYER_SHAPE = LS_SIGMOID_2

    # Normalize the data
    X_train, y_train, X_test, y_test = get_train_test_pd()
    y_train = y_train.to_numpy().reshape(-1, 1)
    y_test = y_test.to_numpy().reshape(-1, 1)

    b_epoch = 0
    b_acc = 0
    seed = np.random.randint(0, 1000000000)
    layers = setup_layers(sigmoid, der_sigmoid, LAYER_SHAPE, seed)

    activations = [None] * N_LAYERS

    for epoch in range(EPOCHS):
        # Forward propagation
        if epoch % STEP_SIZE == 0:
            LEARNING_RATE *= DECAY_RATE

        if (epoch > (EPOCHS / 1.5) and epoch % 5 == 0):
            input_train = X_train
            for i in range(N_LAYERS):
                activations[i], _ = layers[i].forward(input_train)
                input_train = activations[i]
            acc_train = f_r2score(y_train, activations[-1])
                
            input_train = X_test
            for i in range(N_LAYERS):
                activations[i], _ = layers[i].forward(input_train)
                input_train = activations[i]
            acc_test = f_r2score(y_test, activations[-1])

            if (acc_train + acc_test > b_acc):
                b_epoch = epoch
                b_acc = acc_train + acc_test
        train_input = X_train
        for i in range(N_LAYERS):
            activations[i], _ = layers[i].forward(train_input)
            train_input = activations[i]

        for i in reversed(range(N_LAYERS)):
            layers[i].backward(y_train, LEARNING_RATE)

    return seed, b_acc, b_epoch