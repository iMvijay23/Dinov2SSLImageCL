import argparse
import torch
import sys
from torch.utils.tensorboard import SummaryWriter
from dataset_preparation import prepare_dataloaders, get_simple_transforms, get_hard_transforms
from models import prepare_models, freeze_student_except_last_layer
from optimizers_schedulers import build_optimizer, build_scheduler, build_criterion
from train import train
from torchvision import transforms  
from evaluate import evaluate
import numpy as np
import torch

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Train a model using MixUp and CutMix augmentations")
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the model checkpoints')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='Number of workers for data loading')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--save_every', type=int, default=25, help='Save model checkpoints every specified number of steps')
    parser.add_argument('--grad_accum_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--transform_type', choices=['simple', 'hard'], default='simple', help='Choose simple or hard transformations')
    parser.add_argument('--test_split_percent', type=float, default=0.2, help='Percentage of training set to keep aside for the test set')
    parser.add_argument('--model_name', choices=['dinov2', 'vit'], default='dinov2', help='Choose the model architecture: dinov2 or vit')
    parser.add_argument('--num_classes', type=int, default=200, help='Number of output classes')
    parser.add_argument('--optimizer', choices=['sgd', 'adam', 'adamw'], default='adamw', help='Choose the optimizer: sgd, adam, or adamw')
    parser.add_argument('--scheduler', choices=['cosineannealing', 'custom'], default='custom', help='Choose the scheduler: cosineannealing or custom')
    parser.add_argument('--baseline', action='store_false', help='Evaluate the model on the validation set before training')
    parser.add_argument('--evaluate', action='store_false', help='Evaluate and show accuracy, loss on saved model')
    args = parser.parse_args()

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Choose the appropriate transformations
    if args.transform_type == 'simple':
        transform1, transform2 = get_simple_transforms()
    else:
        transform1, transform2 = get_hard_transforms()

    eval_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # DataLoader preparation
    train_loader, eval_loader, test_loader = prepare_dataloaders(transform1, transform2, eval_transform, args.test_split_percent, batch_size=args.batch_size, num_workers=args.num_workers)
    print("Finished preparing the dataloader")
    print()
    print("Preparing student and teacher models")
    # Model preparation
    student_model, teacher_model = prepare_models(args.model_name, num_classes=args.num_classes)
    student_model = freeze_student_except_last_layer(student_model)

    #Evaluate on saved model and break 
    if args.evaluate:
        saved_model_path = "path/to/saved/model"
        student_model.load_state_dict(torch.load(saved_model_path))
        eval_accuracy, eval_loss, _ = evaluate(student_model, test_loader, device)
        print(f"Baseline validation accuracy: {eval_accuracy:.2f}%")
        print(f"Baseline validation loss: {eval_loss:.4f}")
        sys.exit()

    print()
    print("Optimizers, schedulers")
    # Optimizer, scheduler, and criterion
    optimizer = build_optimizer(args.optimizer, student_model)
    T_max = args.epochs * len(train_loader)
    scheduler = build_scheduler(args.scheduler, optimizer, T_max, args.epochs, train_loader)
    criterion = build_criterion()

    # Evaluate the model before training if baseline argument is set
    if args.baseline:
        print()
        print("Evaluating baseline")
        eval_accuracy, eval_loss, _ = evaluate(student_model, test_loader, device)
        print(f"Baseline validation accuracy: {eval_accuracy:.2f}%")
        print(f"Baseline validation loss: {eval_loss:.4f}")

    
    # Initialize TensorBoard writer
    writer = SummaryWriter()
    
    print()
    print("Begining to finetune")
    # Train the model
    train(student_model, teacher_model, train_loader, eval_loader, optimizer, scheduler, device, args, evaluate, writer)

    # Close TensorBoard writer
    writer.close()

    print()
    print("Finetuning is done")

    print()
    print("Evaluating model")
    #Evaluation of the model
    eval_metric, val_loss, correct = evaluate(student_model, eval_loader, device)

    print(f"Finetuned validation accuracy: {eval_metric:.2f}%")
    print(f"Finetuned validation loss: {val_loss:.4f}")

