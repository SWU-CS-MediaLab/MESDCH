# coding: utf-8
# @Time     : 
# @Author   : Godder
# @Github   : https://github.com/WangGodder
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np


mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


train_transform = transforms.Compose([transforms.Resize((224, 224)),
                                # transforms.RandomChoice([
                                #     transforms.RandomRotation([90, 90]),
                                #     transforms.RandomRotation([180, 180]),
                                #     transforms.RandomRotation([270, 270]),
                                #     transforms.RandomRotation([0, 0]),
                                #     transforms.RandomHorizontalFlip(1),
                                #     transforms.RandomVerticalFlip(1)
                                # ]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)])

valid_transform = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean, std)])


class CrossModalTrainBase(Dataset):
    def __init__(self, img_dir: str, img_names: np.ndarray, txt_matrix: np.ndarray, label_matrix: np.ndarray, transform):
        super(CrossModalTrainBase, self).__init__()
        if not os.path.exists(img_dir):
            raise FileExistsError(img_dir + " is not exist")
        self.img_dir = img_dir
        if not os.path.isdir(img_dir):
            raise NotADirectoryError(img_dir + " is not a dir")

        # transform for images
        self.transform = transform
        print(transform)

        self.img_names = img_names
        self.txt = txt_matrix
        self.label = label_matrix

        if img_names.shape[0] != txt_matrix.shape[0] != label_matrix.shape[0]:
            raise ValueError("image name, txt matrix and label matrix must have same num but get %d, %d, %d" %
                             (img_names.shape[0], txt_matrix.shape[0], label_matrix.shape0))

        self.length = self.img_names.shape[0]

        self.random_item = []

        self.img_read = True
        self.txt_read = False

    def read_img(self, item):
        image_url = os.path.join(self.img_dir, self.img_names[item].strip())
        image = Image.open(image_url).convert('RGB')
        image = self.transform(image)
        return image

    def read_txt(self, item):
        pass

    def read_label(self, item):
        pass

    def img_load(self):
        if self.img_read is False:
            self.img_read = True
            self.txt_read = False

    def txt_load(self):
        if self.txt_read is False:
            self.img_read = False
            self.txt_read = True

    def both_load(self):
        self.img_read = False
        self.txt_read = False

    def get_all_label(self):
        return torch.Tensor(self.label)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        pass


class CrossModalValidBase(CrossModalTrainBase):
    """
    cross modal valid base, you only need to init the index and instance of query and retrieval
    """
    def __init__(self, img_dir: str, query_img_names, retrieval_img_names, query_txt_list, retrieval_txt_list,
                 query_label_list, retrieval_label_list, transform=valid_transform, step='query'):
        super(CrossModalValidBase, self).__init__(img_dir, query_img_names, query_txt_list, query_label_list, transform)
        if step is not 'query' and step is not 'retrieval':
            raise ValueError("step only can be one of 'query' and 'retrieval'!!")
        self.is_query = step is 'query'

        self.query_img_names = query_img_names
        self.retrieval_img_names = retrieval_img_names
        self.query_txt = query_txt_list
        self.retrieval_txt = retrieval_txt_list
        self.query_label = query_label_list
        self.retrieval_label = retrieval_label_list

        self.query_num = query_img_names.shape[0]
        self.retrieval_num = retrieval_img_names.shape[0]

    def read_img(self, item):
        if self.is_query:
            image_url = os.path.join(self.img_dir, self.query_img_names[item].strip())
        else:
            image_url = os.path.join(self.img_dir, self.retrieval_img_names[item].strip())
        image = Image.open(image_url).convert('RGB')
        image = self.transform(image)
        return image

    def read_txt(self, item):
        if self.is_query:
            return torch.Tensor(self.query_txt[item][np.newaxis, :, np.newaxis])
        else:
            return torch.Tensor(self.retrieval_txt[item][np.newaxis, :, np.newaxis])

    def read_label(self, item):
        if self.is_query:
            return torch.Tensor(self.query_label[item])
        else:
            return torch.Tensor(self.retrieval_label[item])

    def __len__(self):
        if self.is_query:
            return self.query_num
        return self.retrieval_num

    def __getitem__(self, item):
        if self.img_read:
            img = self.read_img(item)
        if self.txt_read:
            txt = self.read_txt(item)
        label = self.read_label(item)
        index = torch.from_numpy(np.array(item))
        if self.img_read is False:
            return {'index': index, 'txt': txt, 'label': label}
        if self.txt_read is False:
            return {'index': index, 'img': img, 'label': label}
        return {'index': index, 'img': img, 'txt': txt, 'label': label}

    def query(self):
        self.is_query = True

    def retrieval(self):
        self.is_query = False

    def get_step(self):
        if self.is_query:
            return 'query'
        else:
            return 'retrieval'

    def get_all_label(self):
        if self.is_query:
            return torch.Tensor(self.query_label)
        else:
            return torch.Tensor(self.retrieval_label)


__all__ = ['train_transform', 'valid_transform', 'CrossModalTrainBase', 'CrossModalValidBase']


if __name__ == '__main__':
    print(train_transform)