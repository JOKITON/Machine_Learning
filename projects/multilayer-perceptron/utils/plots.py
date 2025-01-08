import matplotlib.pyplot as plt

def plot_acc_epochs(acc_train, acc_test, epochs):
    """ Plot the accuracy over the epochs. """
    epochs = list(range(1, epochs + 2))
    
    # Plotting the graph
    plt.figure(figsize=(20, 12))
    plt.plot(epochs, acc_train, label="Training Acc", color="blue", marker="", linestyle="-")
    plt.plot(epochs, acc_test, label="Validation Acc", color="orange", marker="", linestyle="--")
    plt.ylim(0, 1)

    # Adding labels, title, and legend
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.title("Learning Curves", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)

    # Show the plot
    plt.show()
    
def plot_loss_epochs(loss_train, loss_test, epochs):
    """ Plot the loss over the epochs. """
    epochs = list(range(1, epochs + 2))
    
    # Plotting the graph
    plt.figure(figsize=(20, 12))
    plt.plot(epochs, loss_train, label="Training loss", color="blue", marker="", linestyle="-")
    plt.plot(epochs, loss_test, label="Validation loss", color="orange", marker="", linestyle="--")
    plt.ylim(0, 0.7)

    # Adding labels, title, and legend
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title("Learning Curves", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)

    # Show the plot
    plt.show()