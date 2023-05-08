
import torch
import matplotlib.pyplot as plt

def evaluate_and_collect_same_class(model, loader, device):
    """
    Evaluate the model and collect images belonging to the same class.

    Args:
        model (nn.Module): Model to evaluate.
        loader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to run the evaluation on.
    """
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    same_class_images = {}
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

            # Collect images belonging to the same class
            for i in range(len(predicted)):
                if predicted[i] == labels[i]:
                    if predicted[i].item() not in same_class_images:
                        same_class_images[predicted[i].item()] = []
                    same_class_images[predicted[i].item()].append(images[i].cpu())

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(loader)
    model.train()
    return accuracy, avg_loss, correct, same_class_images


def display_images(images, title, num_images=3):
    """
    Display a specified number of images with the same class label.

    Args:
        images (list): List of images to display.
        title (str): Title for the displayed images.
        num_images (int, optional): Number of images to display. Default: 3.
    """
    fig, axes = plt.subplots(1, num_images, figsize=(9, 3))

    for i, image in enumerate(images[:num_images]):
        axes[i].imshow(image.permute(1, 2, 0))
        axes[i].set_title(title)

    plt.tight_layout()
    plt.show()


# Load the saved model
saved_model_path = "path/to/saved/model"
student_model.load_state_dict(torch.load(saved_model_path))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Evaluate and display images
accuracy, avg_loss, correct, same_class_images = evaluate_and_collect_same_class(student_model, test_loader, device)

num_classes_to_display = 5
counter = 0

class_counts = []

for class_label, images in same_class_images.items():
    if len(images) >= 3:
        display_images(images, f"Class {class_label}")
        class_counts.append(len(images))
        counter += 1

    if counter >= num_classes_to_display:
        break

#plt.hist(class_counts, bins=range(1, max(class_counts) + 1))
#plt.xlabel("Number of Images")
#plt.ylabel("Number of Classes")
#plt.title("Number of Images per Class")
#plt.show()
