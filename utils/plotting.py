import matplotlib.pyplot as plt

def plot_history(history):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure()
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(epochs, history['learning_rate'])

    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.yscale('log')  # Use logarithmic scale for learning rate
    plt.grid(True)
    plt.show()