from torchvision.datasets import * 
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import os
import shutil
from urllib.request import urlretrieve

import torch
import numpy as np
import imageio

from multiprocessing.pool import ThreadPool

from torchvision.transforms import transforms
import random 
from training import *

import logging

def get_loaders(train_config, distribution):

    dataset = TrainingDataset(random_mask_distribution=distribution, all_classes=train_config.all_classes)
    train_data, test_data = dataset.get_dataset(train_config.test_size)

    logging.info(f"Train size {len(train_data)} / Test size {len(test_data)}")

    train_loader, test_loader, shapley_loader = dataset.tuple_to_loader(
        datasets=(train_data, test_data, test_data), 
        batch_sizes=(train_config.train_batch_size, train_config.test_batch_size, 1)
        )

    return train_loader, test_loader, shapley_loader

def load_model_and_loader(train_config):
    class_type = "binary" if not train_config.all_classes else "multiclass"
    mask_type = "masked" if train_config.mask_inputs else "unmasked"
    distrib = str(train_config.mask_mean) + "_" + str(train_config.mask_std)

    model_path = "saved_cnn_model/" + class_type + "/" + mask_type + "_" + distrib
    loader_path = "saved_datasets/" + class_type + "/" + mask_type + "_" + distrib

    model = load_model(model_path).to(train_config.device)
    dataset = TensorDataset(
        torch.from_numpy(np.load("saved_datasets/base/inputs.npy")),
        torch.from_numpy(np.load(loader_path + "/" + "processed.npy")),
        torch.from_numpy(np.load("saved_datasets/base/masks.npy")),
        torch.from_numpy(np.load("saved_datasets/base/labels.npy")),
        )

    return model, DataLoader(dataset, batch_size=1)

class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


class TrainingDataset():

    def __init__(self, random_mask_distribution=None, all_classes=False):

        self.random_mask_distribution = random_mask_distribution
        self.all_classes = all_classes
    
    def merge_images(self, images, indexes):
        
        images = [images[i] for i in indexes]
        return torch.cat([
            torch.cat([images[0], images[1]], dim=-1), 
            torch.cat([images[2], images[3]], dim=-1)], 
            dim=-2)
        
    def transform_imagenet(self, imgs):
        transform = transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224), 
            #transforms.ToTensor(), 
            transforms.Resize(112),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transform_base = transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224),
            transforms.Resize(112),
            ])
        
        imgs = torch.from_numpy(imgs.transpose(0, 3, 1, 2)) / 255
        img = transform(imgs)
        img_base = transform_base(imgs)
        return img_base, img

    def build_4_images(self, dataset):
        
        import pickle
        with open('datasets/miniimagenet/mini-imagenet-cache-val.pkl', 'rb') as f:
            imagenet = pickle.load(f)

        non_dog_keys = [k for k in imagenet["class_dict"].keys() if k not in ["n02091244", "n02114548"]]
        non_dog_indexes = []
        for k in non_dog_keys:
            non_dog_indexes += imagenet["class_dict"][k]

        imagenet = imagenet["image_data"]
        imagenet = imagenet[non_dog_indexes]
        
        imagenet_base, imagenet = self.transform_imagenet(imagenet)

        outputs = []
        for data in dataset:
            indexes = [0, 1, 2, 3]
            random.shuffle(indexes)
            pos_label = indexes.index(0)


            idx = [random.randrange(0, imagenet.shape[0]) for _ in range(3)]

            img_base = [data[0]] + [imagenet_base[i] for i in idx]
            img_base = self.merge_images(img_base, indexes)

            img = [data[1]] + [imagenet[i] for i in idx]
            img = self.merge_images(img, indexes)

            segmentation = [data[2] for i in range(len(indexes))]
            segmentation = self.merge_images(segmentation, indexes)

            outputs.append((img_base, img, segmentation, data[-1], torch.tensor(pos_label)))

        return outputs

    
    def get_dataset(self, test_size=1024):
        
        train_dataset, test_dataset = self.get_oxford_dataset(test_size)
        return (self.build_4_images(train_dataset), self.build_4_images(test_dataset))

    def tuple_to_loader(self, datasets, batch_sizes):

        if isinstance(batch_sizes, tuple) or isinstance(batch_sizes, list):
            assert len(datasets) == len(batch_sizes)
            return [DataLoader(dataset, batch_size=batch_size) for dataset, batch_size in zip(datasets, batch_sizes)]
            
        return DataLoader(datasets, batch_size=batch_sizes)

    def download_url(self, url, filepath):
        directory = os.path.dirname(os.path.abspath(filepath))
        os.makedirs(directory, exist_ok=True)
        if os.path.exists(filepath):
            logging.info("Dataset already exists on the disk. Skipping download.")
            return

        with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=os.path.basename(filepath)) as t:
            urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
            t.total = t.n

    def extract_archive(self, filepath):
        logging.info("Extracting archive")
        extract_dir = os.path.dirname(os.path.abspath(filepath))
        shutil.unpack_archive(filepath, extract_dir)

    def preprocess_mask(self, mask):
        mask = mask.astype(np.float32)
        mask[mask == 1.0] = 1.0
        mask[(mask == 2.0) | (mask == 3.0)] = 0.0

        return mask

    def get_image(self, d, noise):

        transform = transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224), 
            #transforms.ToTensor(), 
            transforms.Resize(112),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transform_base = transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224),
            transforms.Resize(112),
            ])

        dataset_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets/oxford-iiit-pet")
        data = d.split()
        
        if self.all_classes:
            name, label = data[0], int(data[1]) - 1
        else:
            name, label = data[0], int(data[2]) - 1

        img_base = torch.tensor(imageio.imread(os.path.join(dataset_directory, "images/" + name + ".jpg"), pilmode="RGB").transpose(2, 0, 1)).float()

        img = transform(img_base/255)
        img_base = transform_base(img_base/255)

        segmentation = imageio.imread(os.path.join(dataset_directory, "annotations/trimaps/" + name + ".png"), pilmode="L")
        segmentation = transform_base(torch.tensor(self.preprocess_mask(segmentation)).unsqueeze(0))

        if self.random_mask_distribution is not None:
            return (img_base, img * segmentation + (1 - segmentation) * noise.squeeze(0), segmentation, torch.tensor(label))

        return (img_base, img, segmentation, torch.tensor(label))

        

    def get_oxford_dataset(self, test_size):
        
        dataset_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets/oxford-iiit-pet")

        annotation_file = open(os.path.join(dataset_directory, "annotations/list.txt")).readlines()[6:]


        logging.info("Processing dataset")
        if self.random_mask_distribution is not None:
            logging.info(f"Masking with N{self.random_mask_distribution} noise")
        else:
            logging.info(f"Skipping segmentation mask")

        if self.all_classes:
            logging.info("37 classes")
        else:
            logging.info("2 classes")

        if self.random_mask_distribution is not None:
            z = torch.zeros(len(annotation_file), 3, 224, 224)
            noise = torch.normal(
                mean = z + self.random_mask_distribution[0], 
                std = z + self.random_mask_distribution[1]
                ).split(dim=0, split_size=1)
        else:
            noise = [None]*len(annotation_file)

        with ThreadPool(processes=None) as pool:
            outputs = pool.starmap(self.get_image, zip(annotation_file, noise))

        random.shuffle(outputs)
        return outputs[:-test_size], outputs[-test_size:]
