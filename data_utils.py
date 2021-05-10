"""Utilities for generating data etc"""
import os

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

import pdb

import transforms

def generated_transformed_mnist(
    save_path, num_transform=1, num_copy_per_image=1, add_rotation=False, test_data = False
):
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            root="./",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()]
            ),
        ),
        batch_size=64,
        shuffle=True,
    )
    data, label = [], []
    for x, y in train_loader:
        data.append(np.transpose(x.numpy(), (0, 2, 3, 1)))
        label.append(y.numpy())
    dataset = np.concatenate(data, axis=0)
    label = np.concatenate(label, axis=0)

    # Transforming data
    candidates = [shift_right, shift_down, blur, noise, invert]
    # if add_rotation:
    #     candidates.append(rotate)
    target_dataset, target_label = [], []
    for img, y in zip(dataset, label):
        for _ in range(num_copy_per_image):
            ts = np.random.choice(candidates, num_transform, replace=False)
            pdb.set_trace()
            img_t = img.copy()
            for t in ts:
                img_t = t(img_t)
            target_dataset.append(img_t)
            target_label.append(y)
    target_dataset, target_label = np.array(target_dataset), np.array(target_label)

    original_data_name = "mnist_original_data"
    transformed_data_name = "mnist_transformed_data"
    if num_transform > 1:
        original_data_name += "_{}_transform".format(num_transform)
        transformed_data_name += "_{}_transform".format(num_transform)
    if (test_data):
        original_data_name += "_test"
        transformed_data_name += "_test"
    np.save(os.path.join(save_path, original_data_name), dataset)
    np.save(os.path.join(save_path, transformed_data_name), target_dataset)
    np.save(os.path.join(save_path, original_data_name + "_label"), label)
    np.save(os.path.join(save_path, transformed_data_name + "_label"), target_label)

def generated_transformed_color_mnist(
    save_path, num_transform=1, num_copy_per_image=1, add_rotation=False, test_data = False
):
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            root="./",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()]
            ),
        ),
        batch_size=64,
        shuffle=True,
    )
    data, label = [], []
    for x, y in train_loader:
        # pdb.set_trace()
        data.append(np.transpose(transforms.randomly_color(x.numpy()), (0, 2, 3, 1)))
        label.append(y.numpy())
    dataset = np.concatenate(data, axis=0)
    label = np.concatenate(label, axis=0)

    # Transforming data
    candidates = [ 
        # transforms.insta_filter('contrast', 1),
        transforms.insta_filter('hue_rotate', 5),
        shift_right,
        shift_left,
        shift_up,
        # transforms.insta_filter('sepia', 1),
        noise
    ]
    # if add_rotation:
    #     candidates.append(rotate)
    target_dataset, target_label = [], []
    for img, y in zip(dataset, label):
        for _ in range(num_copy_per_image):
            ts = np.random.choice(candidates, num_transform, replace=False)
            # pdb.set_trace()
            img_t = img.copy()
            for t in ts:
                img_t = t(img_t)
            target_dataset.append(img_t)
            target_label.append(y)
    target_dataset, target_label = np.array(target_dataset), np.array(target_label)

    original_data_name = "color_mnist_original_data"
    transformed_data_name = "color_mnist_transformed_data"
    if num_transform > 1:
        original_data_name += "_{}_transform".format(num_transform)
        transformed_data_name += "_{}_transform".format(num_transform)
    if (test_data):
        original_data_name += "_test"
        transformed_data_name += "_test"
    np.save(os.path.join(save_path, original_data_name), dataset)
    np.save(os.path.join(save_path, transformed_data_name), target_dataset)
    np.save(os.path.join(save_path, original_data_name + "_label"), label)
    np.save(os.path.join(save_path, transformed_data_name + "_label"), target_label)


def shift(image, direction):
    axis = 0 if direction[1] == 0 else 1
    amount = 3 if direction[1] == 0 else 5
    amount *= -direction[axis]
    new_image = np.roll(image, amount, axis)
    if (axis == 0):
        if (amount < 0):
            new_image[amount:] = 0
        else:
            new_image[:amount] = 0
    else:
        if (amount < 0):
            new_image[:,amount:] = 0
        else:
            new_image[:,:amount] = 0
    # f, axarr = plt.subplots(nrows=1,ncols=2)
    # plt.sca(axarr[0]); 
    # plt.imshow(image[:,:,0]); plt.title('Source')
    # plt.sca(axarr[1]); 
    # plt.imshow(new_image[:,:,0]); plt.title('Target')
    # plt.savefig('image1.png')
    # plt.clf()
    # pdb.set_trace()
    return new_image



def shift_right(image):
    return shift(image.copy(), [0, -1])


def shift_left(image):
    return shift(image.copy(), [0, 1])


def shift_up(image):
    return shift(image.copy(), [1, 0])


def shift_down(image):
    return shift(image.copy(), [-1, 0])

def blur(image):
    new_image = ndimage.gaussian_filter(image.copy(), 1)
    return new_image


def invert(image):
    image = image.copy()
    if np.max(image) > 1.0:
        return 255.0 - image
    else:
        return 1.0 - image


def noise(image):
    image = image.copy()
    scale = 1.0 if np.max(image) <= 1.0 else 255.0
    added_noise = np.random.binomial(1, 0.05, size=image.shape) * scale
    image += added_noise
    clipped = np.clip(image, 0.0, scale)
    return clipped


def rotate(image):
    image = image.copy()
    return np.transpose(image, [1, 0, 2])


# ========================================================================


def generate_supervised_training_data(args, experts, image_per_expert=10000):
    def get_loader(train=True):
        return torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                root="./",
                train=train,
                download=True,
                transform=torchvision.transforms.Compose(
                    [torchvision.transforms.ToTensor()]
                ),
            ),
            batch_size=64,
            shuffle=True,
            drop_last=True,
        )

    train_loader = get_loader()
    num_batch = image_per_expert // 64
    train_data, train_label = [], []
    for t in experts:
        for _ in num_batch:
            try:
                x, y = next(train_loader)
            except StopIteration:
                train_loader = get_loader()
                x, y = next(train_loader)
            x = x.to(args.device)
            x_t = t(x)
            train_data.append(np.transpose(x_t.cpu().numpy(), (0, 2, 3, 1)))
            train_label.append(y.cpu().numpy())

    train_data = np.concatenate(train_data, axis=0)

def visualize_transforms():
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            root="./",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()]
            ),
        ),
        batch_size=64,
        shuffle=True,
    )
    data, label = [], []
    for x, y in train_loader:
        data.append(np.transpose(x.numpy(), (0, 2, 3, 1)))
        label.append(y.numpy())
    dataset = np.concatenate(data, axis=0)
    label = np.concatenate(label, axis=0)

    # Transforming data
    candidates = [shift_right, shift_down, blur, noise, invert]
    # if add_rotation:
    #     candidates.append(rotate)
    img = dataset[0]
    img_t = img.copy()
    # f, axarr = plt.subplots(nrows=1,ncols=6, figsize=(10, 2))
    # axarr[0].axis('off')
    # plt.sca(axarr[0])
    plt.axis('off')
    # plt.tight_layout()
    plt.imshow(img_t[:,:,0]); plt.title('original')
    plt.savefig('image-1.png')
    for i, c in enumerate(candidates):
        plt.clf()
        img_t = c(img_t)
        plt.axis('off')
        # axarr[i+1].axis('off')
        # plt.sca(axarr[i+1]); 
        plt.imshow(img_t[:,:,0]); plt.title(c.__name__)
        plt.savefig(f'image{i}.png')

    # plt.savefig('image0.png')
    plt.clf()
    # pdb.set_trace()

    
if __name__ == "__main__":
    # visualize_transforms()
    generated_transformed_color_mnist('data', num_transform=1, add_rotation=True, test_data=False)
            
