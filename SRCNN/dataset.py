
import os


from PIL import Image

from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, Resize, ToPILImage

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['bmp', '.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


class DatasetFromfolder(Dataset):
    def __init__(self, path):
        super(DatasetFromfolder, self).__init__()
        self.filenames = []
        folders = os.listdir(path)
        for f in folders:
            self.filenames.append(path + f)
        self.data_transform = Compose([RandomCrop([33, 33]), ToTensor()])
        self.data_transform_PIL = Compose([ToPILImage()])

    def __getitem__(self, index):
        w = h = 33
        image = Image.open(self.filenames[index])
        image, _cb, _cr = image.convert('YCbCr').split()
        image = self.data_transform(image)
        result_image = image

        resize_image = self.data_transform_PIL(image)
        resize_image = resize_image.resize((int(w / 3), int(h / 3)))
        resize_image = resize_image.resize((w, h), Image.BICUBIC)
        resize_image = self.data_transform(resize_image)

        return result_image, resize_image

    def __len__(self):
        return len(self.filenames)
