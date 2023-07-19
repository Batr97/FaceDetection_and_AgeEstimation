import os
import cv2 as cv
import numpy as np
import albumentations as augs


def open_files(path):
    _, _, files = next(os.walk(path))
    return files


def generation_new_imgs(files, path, save_path, num_generations, transformations):           
        for num in range(num_generations):
            for item in files:
                filename = path + item
                image = cv.imread(filename)
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                augmented = transformations(image=image)
                augmented_image = augmented['image']
                image_name = f'aug_{num}_{item}'
                image_path = os.path.join(save_path, image_name)
                cv.imwrite(image_path, cv.cvtColor(augmented_image.astype(np.uint8), cv.COLOR_RGB2BGR))


generate_new_images = True # if we need to generate new images by applying augmentations to them
if generate_new_images:
    
    transformations = augs.Compose([
        augs.Rotate(15),
        augs.OneOf([
            augs.CLAHE(),
            augs.RandomGamma(),
            augs.RandomBrightnessContrast()]),
        augs.OneOf([
            augs.GaussianBlur(),
            augs.MedianBlur(blur_limit=3),
            augs.MotionBlur()], p=0.2),
        augs.OneOf([
            augs.GaussNoise(),
            augs.ISONoise(),]),
        augs.HorizontalFlip(p=0.2),
    ])
    
    files = open_files('data/UTKFace_Dataset/')
    path = 'data/UTKFace_Dataset/'
    save_path = 'data/UTKFace_Dataset_augmented/'
    generation_new_imgs(files, path, save_path, 1, transformations)