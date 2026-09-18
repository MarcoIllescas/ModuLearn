import torch


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train() # poner el modelo en modo entrenamiento

    total_loss = 0.0
    total_samples = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device) #mover los datos al dispositivo
        labels = labels.to(device) #mover las etiquetas al dispositivo 

        outputs = model(inputs) #pasar los logits por el modelo
        loss = criterion(outputs, labels) #calcular la pérdida

        optimizer.zero_grad() #reiniciar los gradientes
        loss.backward() #calcular los gradientes
        optimizer.step() #actualizar los pesos

        total_loss += loss.item() * inputs.size(0) #acumular la pérdida (tamaño del batch)
        total_samples += labels.size(0) #acumular el número de muestras

    avg_loss = total_loss / total_samples #retornar la pérdida promedio por muestra
    return avg_loss

def evaluate(model, dataloader, criterion, device):
    model.eval() # poner el modelo en modo evaluación

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad(): #desactivar el cálculo de gradientes
        for inputs, labels in dataloader:
            inputs = inputs.to(device) #mover los datos al dispositivo
            labels = labels.to(device) #mover las etiquetas al dispositivo 

            outputs = model(inputs) #pasar los logits por el modelo
            loss = criterion(outputs, labels) #calcular la pérdida


            total_loss += loss.item() * inputs.size(0) #acumular la pérdida (tamaño del batch)
            total_samples += labels.size(0) #acumular el número de muestras

    avg_loss = total_loss / total_samples #retornar la pérdida promedio por muestra
    return avg_loss

def fit(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, early_stopping=None, scheduler=None):
    history = {'train_loss': [], 'val_loss': [], "learning_rate": []}

    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(current_lr)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Learning Rate: {current_lr:.2e}')

        if scheduler is not None:
            scheduler.step(val_loss)  # Update the learning rate based on validation loss
            
        if early_stopping is not None:
            early_stopping(model, val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

    if early_stopping is not None:
        early_stopping.restore_best_model(model)

    return history