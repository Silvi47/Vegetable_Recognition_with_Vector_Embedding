from typing import Callable
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from quaterion.dataset import SimilarityGroupSample

class preprocessing(Dataset):
    def __init__(self, dataset: Dataset, transform: Callable):
        self._dataset = dataset
        self._transform = transform

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index) -> SimilarityGroupSample:
        image, label = self._dataset[index]
        image = self._transform(image)

        return SimilarityGroupSample(obj=image, group=label)

def create_transform(input_size: int):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    return transform