import cv2
import os
import PIL
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

PATH_TO_DATASET = "img_align_celeba"

# -------------------------------------------------------------
# only for testing: displaying some (few) pictures
# -------------------------------------------------------------
# PATH_TO_DATASET = "img_align_celeba_test"
# -------------------------------------------------------------

PATH_TO_DATASET_WITH_CLASSES = PATH_TO_DATASET + "/celebrities"

def load_dataset():
    images = []

    for filename in os.listdir(PATH_TO_DATASET_WITH_CLASSES):
        img = cv2.imread(os.path.join(PATH_TO_DATASET_WITH_CLASSES, filename))

        # convert to tensor + normalize it to [0, 1]
        transform = torchvision.transforms.ToTensor()
        img = transform(img)

        if img is not None:
            images.append(img)

    return images

# -------------------------------------------------------------
# only for testing: displaying some (few) pictures
# -------------------------------------------------------------
# def show_images(images, labels, class_names):
#     fig, axes = plt.subplots(1, len(images), figsize=(12, 3))
#     for i, (image, label) in enumerate(zip(images, labels)):
#         image = image.permute(1, 2, 0).numpy()  # CHW to HWC
#         axes[i].imshow(image)
#         axes[i].set_title(class_names[label])
#         axes[i].axis('off')
#     plt.tight_layout()
#     plt.show()
# -------------------------------------------------------------

def image_augmentation():
    # CelebA dataset from:  https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    images_loaded_dataset = load_dataset()
    print(f"image_augmentation(): {len(images_loaded_dataset)} images loaded")

    # based on tutorial: https://www.kaggle.com/code/mohamedmustafa/7-data-augmentation-on-images-using-pytorch
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ColorJitter(brightness=(0.1, 0.5), contrast=0.5, saturation=0, hue=0.5),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomRotation(20),
        torchvision.transforms.RandomCrop(size=(112, 112)),
        torchvision.transforms.ToTensor()
        # TODO: normalize?
    ])

    dataset = torchvision.datasets.ImageFolder(PATH_TO_DATASET, transform=transforms)

    # ---------------------------------------------------------------------------
    # only for testing: displaying some (few) pictures
    # ---------------------------------------------------------------------------
    # note: batch_size = 5 for 5 test pictures
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True)
    # transformed_images, labels = next(iter(dataloader))
    # class_names = dataset.classes
    # show_images(transformed_images, labels, class_names)
    # ---------------------------------------------------------------------------

    # add images with augmentation to list with default images
    transformed_images = [image for image, label in dataset]

    all_images = images_loaded_dataset + transformed_images
    print(f"image_augmentation(): {len(all_images)} images after transformation")

    return all_images

def main():
    # all_images_of_dataset is a list of tensors (image = saved as tensor)
    all_images_of_dataset = image_augmentation()


if __name__ == "__main__":
    main()

