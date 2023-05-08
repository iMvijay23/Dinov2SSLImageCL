import torch
from tqdm import tqdm


def evaluate(model, loader, device):
    """
    Evaluation function to calculate loss and accuracy on Val/test dataset
    Args:
        model (nn.Module): model to be evaluated on the give dataset
        loader (DataLoader): Validation/Test dataloader to evaluate the model on.
        device (torch.device): The device (CPU/GPU) to perform the evalutation on. 
    """
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            predicted = torch.argmax(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            running_loss += loss.item()

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(loader)
    model.train()
    return accuracy, avg_loss, correct
