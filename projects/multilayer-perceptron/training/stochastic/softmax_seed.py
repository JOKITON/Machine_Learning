""" Program that creates a Multilayer Perceptron model to detect type of cancer cells. """

def init_st_soft():
    from config import LEARNING_RATE, STEP_SIZE, DECAY_RATE, CONVERGENCE_THRESHOLD
    from config import N_FEATURES, EPOCHS_STOCHASTIC_2, LS_SOFTMAX_1, N_LAYERS, BATCH_SIZE, DF_UTILS
    import numpy as np
    from preprocessing import get_train_test_pd
    from batch import get_stochastic
    from activations import softmax, der_softmax
    from setup import setup_layers
    from loss import f_r2score

    EPOCHS = EPOCHS_STOCHASTIC_2
    LAYER_SHAPE = LS_SOFTMAX_1

    # Normalize the data
    X_train, y_train, X_test, y_test = get_train_test_pd()
    y_train = y_train.to_numpy().reshape(-1, 1)
    y_test = y_test.to_numpy().reshape(-1, 1)
    
    b_epoch = 0
    b_acc = 0
    seed = np.random.randint(0, 1000000000)
    layers = setup_layers(softmax, der_softmax, LAYER_SHAPE, seed)

    activations = [None] * N_LAYERS

    y_train_softmax = np.zeros((y_train.shape[0], 2))
    y_train_softmax[np.arange(y_train.shape[0]), y_train.flatten()] = 1

    y_test_softmax = np.zeros((y_test.shape[0], 2))
    y_test_softmax[np.arange(y_test.shape[0]), y_test.flatten()] = 1

    for epoch in range(EPOCHS):
        # Forward propagation
        if epoch % STEP_SIZE == 0:
            LEARNING_RATE *= DECAY_RATE
        
        if (epoch > (EPOCHS / 1.5) and epoch % 5 == 0):
            input_train = X_train
            for i in range(N_LAYERS):
                activations[i], _ = layers[i].forward(input_train)
                input_train = activations[i]
            acc_train = f_r2score(y_train_softmax, activations[-1])
                
            input_train = X_test
            for i in range(N_LAYERS):
                activations[i], _ = layers[i].forward(input_train)
                input_train = activations[i]
            acc_test = f_r2score(y_test_softmax, activations[-1])
            
            if (acc_train + acc_test > b_acc):
                b_epoch = epoch
                b_acc = acc_train + acc_test

        train_x, train_y = get_stochastic(X_train, y_train)
        for i in range(N_LAYERS):
            activations[i], _ = layers[i].forward(train_x)
            train_x = activations[i]

        for i in reversed(range(N_LAYERS)):
            if (i == N_LAYERS - 1):
                y_true_one_hot = np.zeros((train_y.shape[0], 2))
                y_true_one_hot[np.arange(train_y.shape[0]), train_y.flatten()] = 1
                input_y = y_true_one_hot
            else:
                input_y = train_y
            layers[i].backward(input_y, LEARNING_RATE)
    
    return seed, b_acc, b_epoch
