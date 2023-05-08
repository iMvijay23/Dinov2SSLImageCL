
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from timm import create_model
from PIL import Image
from datasets import load_dataset


def train(student_model, teacher_model, train_loader, eval_loader, optimizer, scheduler, device, args, evaluate, writer):
    """
    Trainer function for finetuning the student model using the teacher model. The teacher model is updated with an exponential moving average of the student model parameters.

    Args:
        student_model (nn.Module): The student model to be finetuned.
        teacher_model (nn.Module): The teacher model used for distillation.
        train_loader (DataLoader): DataLoader for the training dataset.
        eval_loader (DataLoader): DataLoader for the evaluation dataset.
        optimizer (Optimizer): The optimizer for the student model.
        scheduler (LRScheduler): The learning rate scheduler for the optimizer.
        device (torch.device): The device (CPU/GPU) to perform the training on.
        args (argparse.Namespace): Parsed command-line arguments.
        evaluate (function): Function to evaluate the student model on the evaluation dataset.
        writer (SummaryWriter): TensorBoard writer for logging training metrics.
    """
    #writer = SummaryWriter()
    save_path = args.save_path

    teacher_model.to(device)
    student_model.to(device)

    # Training loop
    best_eval_metric = -1  

    gradient_accumulation_steps = args.grad_accum_steps
    scaler = GradScaler()

    for epoch in tqdm(range(args.epochs)):
        student_model.train()
        teacher_model.train()

        optimizer.zero_grad()

        for i, (images1, images2, _) in enumerate(train_loader):
            images1 = images1.to(device)
            images2 = images2.to(device)

            with autocast():
                # Get feature representations from student model
                student_repr1 = student_model(images1)
                student_repr2 = student_model(images2)
                student_repr = nn.functional.normalize(student_repr1 + student_repr2, dim=1)

                # Get feature representations from teacher model
                with torch.no_grad():
                    teacher_repr1 = teacher_model(images1)
                    teacher_repr2 = teacher_model(images2)
                    teacher_repr = nn.functional.normalize(teacher_repr1 + teacher_repr2, dim=1)

                # Compute the temperature-scaled dot product similarity matrix
                temperature = 0.1
                sim_matrix = torch.matmul(student_repr, teacher_repr.t().detach()) / temperature

                # Compute DINO loss (KL divergence between student and teacher outputs)
                per_example_losses = torch.nn.functional.kl_div(torch.log_softmax(sim_matrix, dim=-1), torch.softmax(sim_matrix.t(), dim=-1).detach(), reduction="none").sum(dim=1)
                loss = per_example_losses.mean()
                # TensorBoard logging
                writer.add_scalars(main_tag="TrainingLoss",
                           tag_scalar_dict={"train_loss": loss},
                           global_step=i+1/len(train_loader))

            loss.backward()

            if (i + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

                # Update teacher model with exponential moving average of student model parameters
                with torch.no_grad():
                    for student_params, teacher_params in zip(student_model.parameters(), teacher_model.parameters()):
                        teacher_params.data.mul_(args.ema_decay).add_(student_params.data, alpha=1 - args.ema_decay)

                scheduler.step()

            if i % args.save_every == 0:
                print(f"Epoch: {epoch+1}/{args.epochs}, Batch: {i+1}/{len(train_loader)}, Train Loss: {loss.item()}")
                torch.save(student_model.state_dict(), f"{save_path}/ViT_DINO_student_epoch_{epoch+1}_Tinyimagenet.pt")

            # Evaluate and save best model every args.save_every steps
            if (i + 1) % args.save_every == 0:
                eval_metric, val_loss, count = evaluate(student_model, eval_loader, device)

                # Add validation loss and accuracy to TensorBoard writer
                print(f"Epoch: {epoch+1}/{args.epochs}, Batch: {i+1}/{len(train_loader)}, Acc Valid: {eval_metric}, Val loss:{val_loss}")
                writer.add_scalars(main_tag="ValLoss",
                           tag_scalar_dict={"val_loss": val_loss},
                           global_step=i+1/len(train_loader))
                writer.add_scalars(main_tag="ValACC",
                           tag_scalar_dict={"val_accuracy": eval_metric},
                           global_step=i+1/len(train_loader))
                if eval_metric > best_eval_metric:
                    best_eval_metric = eval_metric
                    # Save student model checkpoint
                    torch.save(student_model.state_dict(), f"{save_path}/ViT_DINO_student_epoch_{epoch+1}_best_tinyimagenet.pt")
                # Save student model checkpoint
        torch.save(student_model.state_dict(), f"{save_path}/ViT_DINO_student_epoch_{epoch+1}_tinyimagenet.pt")

    # Close TensorBoard writer
    writer.close()

