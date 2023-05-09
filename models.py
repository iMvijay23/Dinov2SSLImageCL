
import torch
import torch.nn as nn
from timm import create_model

def prepare_models(model_name, num_classes=200, pretrained=True):
    """
    Function to prepare student and teacher models for DINOv2 finetuning approach of distillation.
    Args:
        num_classes (int): Total number of classes in output layer.
    """
    if model_name == "dinov2":
        student_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_lc', pretrained=pretrained)
        teacher_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_lc', pretrained=pretrained)
        student_model.linear_head = nn.Linear(in_features=1920, out_features=num_classes, bias=True)
        teacher_model.linear_head = nn.Linear(in_features=1920, out_features=num_classes, bias=True)
    elif model_name == "dinov2base":
        student_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_lc', pretrained=pretrained)
        teacher_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_lc', pretrained=pretrained)
        student_model.linear_head = nn.Linear(in_features=3840, out_features=num_classes, bias=True)
        teacher_model.linear_head = nn.Linear(in_features=3840, out_features=num_classes, bias=True)
    elif model_name == "vit":
        student_model = create_model("vit_base_patch32_224_in21k", pretrained=pretrained)
        teacher_model = create_model("vit_base_patch32_224_in21k", pretrained=pretrained)
        student_model.head = nn.Linear(in_features=768, out_features=num_classes, bias=True)
        teacher_model.head = nn.Linear(in_features=768, out_features=num_classes, bias=True)
    else:
        raise ValueError("Invalid model name")

    teacher_model.load_state_dict(student_model.state_dict())

    return student_model, teacher_model

def freeze_student_except_last_layer(student_model):
    """
    Function to freeze all student paramters except the last linear head layer.
    Args:
        student_model (nn.Module): Student model to be finetuned using teacher model with distillation approach
    """
    for name, param in student_model.named_parameters():
        if 'head' not in name and 'linear_head' not in name:
            param.requires_grad = False

    return student_model
