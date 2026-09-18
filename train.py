import torch
import torch.nn as nn
from utils.device import get_device
from datasets.mnist import get_mnist_dataset
from models.mlp import MLP
from models.cnn import CNN
from engine.trainer import fit, evaluate
from utils.plotting import plot_history
from callbacks.early_stopping import EarlyStopping

def main() -> None:
    #hiperparámetros
    num_epochs = 100 # Callback - EarlyStopping -> REGULARIZATION
    learning_rate = 0.1 # -> Controla la magnitud de los cambios
    # StepLR -> Cada n epochs, reduce el learning rate por un factor gamma
    # ExponentialLR -> Reduce el learning rate exponencialmente
    # ReduceLROnPlateau -> Reduce el learning rate cuando la métrica de validación deja de mejorar

    batch_size = 64
    patience = 6
    min_delta = 1e-3
    weight_decay = 1e-4 # -> Regularization (L2) -> Penaliza los pesos grandes


    device = get_device() #obtener el dispositivo a usar
    print(f'Training on: {device}')

    train_loader, val_loader, test_loader = get_mnist_dataset(
        data_dir='./data', 
        batch_size=batch_size,
        val_split=0.2
        ) #obtener los dataloaders

    ## Model
    model = CNN()
    # model = MLP()
    model = model.to(device) #mover el modelo al dispositivo

    #entrenamiento
    criterion = nn.CrossEntropyLoss() #función de pérdida
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # -> Determina como actualizamos parametros
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2) # -> Cambia el LR durante el entrenamiento

    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    history = fit(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        device, 
        num_epochs,
        early_stopping,
        scheduler
    )

    torch.save(model.state_dict(), 'artifacts/best_model.pth') 

    test_loss = evaluate(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}')
    plot_history(history)
    

if __name__ == '__main__':
    main()