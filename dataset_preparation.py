import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from torch.utils.data.dataset import Dataset


transform1simple= transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(224),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
])

transform2simple = transforms.Compose([
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(60),
    transforms.Resize(224),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
])

transform1 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(224),
    transforms.RandomRotation(30),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomCrop(size=224, padding=4, pad_if_needed=True),
    transforms.ToTensor(),
])

transform2 = transforms.Compose([
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(60),
    transforms.Resize(224),
    transforms.RandomCrop(size=224, padding=4, pad_if_needed=True),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
])




class HFTinyImageNetDataseteval(Dataset):
    """
    Custom dataset class for evaluating Hugging Face TinyImageNet dataset.
    """
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __getitem__(self, index):
        image = self.hf_dataset[index]['image']
        label = self.hf_dataset[index]["label"]
        if image.mode == 'L':
          image = image.convert('RGB')
        elif image.mode == 'RGBA':
          image = image.convert('RGB')

        transform = transforms.Compose([
            transforms.Resize((224)), 
            transforms.ToTensor(), 
        ])
        image = transform(image)
        return image, label

    def __len__(self):
        return len(self.hf_dataset)

class MyDataset(Dataset):
    """
    Custom dataset class for training dataset using two transform augmentations.
    """
    def __init__(self, data, transform1=None, transform2=None):
        self.data = data
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]["image"]
        label = torch.tensor(self.data[idx]['label'], dtype=torch.long)

        if image.mode == 'L':
          image = image.convert('RGB')
        elif image.mode == 'RGBA':
          image = image.convert('RGB')
        
        if self.transform1 is not None:
            image1 = self.transform1(image)
        
        if self.transform2 is not None:
            image2 = self.transform2(image)

        return image1, image2, label

def custom_collate(batch):
    """
    Custom collate function to handle batches with MixUp and CutMix augmentations.

    Args:
        batch (list): List of tuples containing images and labels.
    """
    images1 = torch.stack([item[0] for item in batch])
    images2 = torch.stack([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch])
    return images1, images2, labels

def prepare_dataloaders(transform1, transform2, eval_transform, test_split_percent, batch_size, num_workers):
    """
    Prepare data loaders for the training, evaluation, and test sets.

    Args:
        transform1 (transforms.Compose): Transformations applied to the first image in the dataset.
        transform2 (transforms.Compose): Transformations applied to the second image in the dataset.
        eval_transform (transforms.Compose): Transformations applied to the evaluation dataset.
        test_split_percent (float): Percentage of the data to be used as the test set.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.
    """
    # Load the dataset
    dataset = load_dataset('Maysee/tiny-imagenet', split='train')
    train_size = int((1-test_split_percent) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset = MyDataset(dataset.select(range(train_size)), transform1, transform2)
    test_dataset = dataset.select(range(train_size, train_size + test_size))

    # Prepare data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=custom_collate)
    eval_dataset = load_dataset('Maysee/tiny-imagenet', split='valid')
    eval_dataset = HFTinyImageNetDataseteval(eval_dataset, transform=eval_transform)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataset = HFTinyImageNetDataseteval(test_dataset, transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, eval_loader, test_loader



def get_simple_transforms():
    return transform1simple, transform2simple

def get_hard_transforms():
    return transform1, transform2
