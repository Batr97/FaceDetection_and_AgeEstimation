import os
import cv2 as cv
import numpy as np
import albumentations as augs
from retinaface import RetinaFace


def open_files(path):
    _, _, files = next(os.walk(path))
    return files


def extract_imgs_with_faces(files):
    imgs = []
    for item in files:
        if 'face' in item:
            imgs.append(item)
    return imgs


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


files = open_files('data/appa-real-release/valid/')
new_files_valid = extract_imgs_with_faces(files)

files = open_files('data/appa-real-release/train/')
new_files_train = extract_imgs_with_faces(files)


def save_cropped_imgs(new_files, path):
    for item in new_files:
        faces = RetinaFace.extract_faces(img_path = path + item, align = True)
        if len(faces) > 0:
            cv.imwrite(f'data/appa-real-release/cropped_images/{item.split("_")[0]}', cv.cvtColor(faces[0].astype(np.uint8), cv.COLOR_RGB2BGR))
    print(f'saving done from {path.split("/")[-1]}!')
    

need_crop = False # if we need to create new dataset from existing by extracting faces from whole images (already done)
if need_crop:
    save_cropped_imgs(new_files_valid, 'data/appa-real-release/valid/')
    save_cropped_imgs(new_files_train, 'data/appa-real-release/train/')


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

    files = open_files('data/appa-real-release/cropped_images/')
    path = 'data/appa-real-release/cropped_images/'
    save_path = 'data/appa-real-release/augmented_imgs/'
    generation_new_imgs(files, path, save_path, 1, transformations)
