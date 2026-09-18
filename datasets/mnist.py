import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_mnist_dataset(data_dir, batch_size=64, val_split=0.2):
    # Data augmentation and normalization for training
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]) # Transformar a tensor

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]) # Transformar a tensor
    
    full_train_dataset = datasets.MNIST(
        data_dir, #dirección de los datos
        train=True, #que queremos descargar, si entrenamiento o validación
        download=True, #si lo queremos descargar
        transform=train_transform,    #transformación a aplicar
    )

    num_train = len(full_train_dataset)
    num_val = int(val_split * num_train)
    num_train -= num_val

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [num_train, num_val]
    )

    test_dataset = datasets.MNIST(
        data_dir,
        train=False,
        download=False,
        transform=eval_transform
    )

    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size

    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(
        full_train_dataset, #dataset de entrenamiento
        batch_size=batch_size, #tamaño del batch
        shuffle=True #mezclar los datos
    )

    val_loader = DataLoader(
        val_dataset, #dataset de validación
        batch_size=batch_size, #tamaño del batch
        shuffle=False #no mezclar los datos
    )

    test_loader = DataLoader(
        test_dataset, #dataset de prueba
        batch_size=batch_size, #tamaño del batch
        shuffle=False #no mezclar los datos
    )

    return train_loader, val_loader, test_loader