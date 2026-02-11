import os
import glob
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# ================= CONFIGURATION =================
BASE_DIR = 'modis_dataset_brazil'
IMG_SIZE = 64
BATCH_SIZE = 64


class SatelliteDataset(Dataset):
    def __init__(self, file_list, label, transform=None):
        """
        Args:
            file_list (list): List of file paths.
            label (int): 0 for Normal, 1 for Fire.
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.file_list = file_list
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]

        # Open image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Fallback for corrupted images (though your clean script should have caught them)
            # Create a black image
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        # Return image and label
        return image, torch.tensor(self.label, dtype=torch.float32)


def get_data_loaders(base_dir, img_size=64, batch_size=64):
    """
    Prepares the Train (Normal Only) and Test (Normal + Fire) loaders.
    """

    # 1. Define Transforms
    # VAE-WGANs usually prefer input in range [-1, 1] for Tanh activation
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # Converts to [0, 1]
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Converts to [-1, 1]
    ])

    # 2. Get File Lists
    fire_dir = os.path.join(base_dir, 'fire_anomalies')
    normal_dir = os.path.join(base_dir, 'normal_reference')

    all_fire = glob.glob(os.path.join(fire_dir, "*.png"))
    all_normal = glob.glob(os.path.join(normal_dir, "*.png"))

    print(f"Total Normal: {len(all_normal)}")
    print(f"Total Fire: {len(all_fire)}")

    # 3. Create Splits
    # TRAINING SET: Needs ONLY Normal data.
    # Let's use 80% of normal data for training.
    train_normal, test_normal = train_test_split(all_normal, test_size=0.2, random_state=42)

    # TEST SET: Needs a mix of Normal (the 20% left over) and Fire.
    # To keep the test set balanced (50/50), we select the same amount of fire images.
    num_test_samples = len(test_normal)
    test_fire = all_fire[:num_test_samples]  # Take top N fire images

    print(f"--- Split Summary ---")
    print(f"Train Set (Normal Only): {len(train_normal)} images")
    print(f"Test Set (Normal):       {len(test_normal)} images")
    print(f"Test Set (Fire):         {len(test_fire)} images")
    print(f"Total Test Set:          {len(test_normal) + len(test_fire)} images")

    # 4. Create Dataset Objects
    train_dataset = SatelliteDataset(train_normal, label=0, transform=transform)

    # For testing, we combine normal and fire
    test_dataset_normal = SatelliteDataset(test_normal, label=0, transform=transform)
    test_dataset_fire = SatelliteDataset(test_fire, label=1, transform=transform)
    test_dataset = torch.utils.data.ConcatDataset([test_dataset_normal, test_dataset_fire])

    # 5. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Important for training
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle for evaluation
        num_workers=2
    )

    return train_loader, test_loader

# train_loader, test_loader = get_data_loaders(BASE_DIR)