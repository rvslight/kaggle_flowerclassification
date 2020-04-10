import cv2
import numpy as np
import random
from albumentations import (
    OneOf, Compose,
    Flip, ShiftScaleRotate, GridDistortion, ElasticTransform,
    RandomGamma, RandomContrast, RandomBrightness, RandomBrightnessContrast,
    Blur, MedianBlur, MotionBlur,
    CLAHE, IAASharpen, GaussNoise,
    HueSaturationValue, RGBShift, IAAAdditiveGaussianNoise)


def aug(p=1.0):
    return Compose([
        Flip(p=0.75),  # ok
    ], p=p)



def just_brightness_flip(p=1.0):
    return Compose([
        Flip(p=0.75),  # ok
        # RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0, brightness_by_max=False),
        # RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        # ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, border_mode=cv2.BORDER_CONSTANT, p=0.2),
        # OneOf([
        #     Blur(blur_limit=5, p=1.0),
        #     MedianBlur(blur_limit=5, p=1.0),
        #     MotionBlur(p=1.0),
        # ], p=0.2),
        # OneOf([
        #     HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=1.0),
        #     RGBShift(p=1.0)
        # ], p=0.1),
        # GaussNoise(p=0.1),

        # OneOf([
        #     GridDistortion(p=1.0),
        #     ElasticTransform(p=1.0)
        # ], p=0.2),
        # OneOf([
        #     CLAHE(p=1.0),
        #     IAASharpen(p=1.0),
        # ], p=0.2)
    ], p=p)

def strong_aug(p=1.0):
    return Compose([
        Flip(p=0.75),  # ok
        # RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        # RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0, brightness_by_max=False),
        # # RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        # ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, border_mode=cv2.BORDER_CONSTANT, p=0.2),
        # OneOf([
        #     Blur(blur_limit=5, p=1.0),
        #     MedianBlur(blur_limit=5, p=1.0),
        #     MotionBlur(p=1.0),
        # ], p=0.2),
        # OneOf([
        #     HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=1.0),
        #     RGBShift(p=1.0)
        # ], p=0.1),
        # GaussNoise(p=0.1),

        # OneOf([
        #     GridDistortion(p=1.0),
        #     ElasticTransform(p=1.0)
        # ], p=0.2),
        # OneOf([
        #     CLAHE(p=1.0),
        #     IAASharpen(p=1.0),
        # ], p=0.2)
    ], p=p)

def strong_aug_no_bright(p=1.0):
    return Compose([
        Flip(p=0.75),  # ok
        # RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, border_mode=cv2.BORDER_CONSTANT, p=0.2),
        OneOf([
            Blur(blur_limit=5, p=1.0),
            MedianBlur(blur_limit=5, p=1.0),
            MotionBlur(p=1.0),
        ], p=0.2),
        OneOf([
            HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=1.0),
            RGBShift(p=1.0)
        ], p=0.1),
        GaussNoise(p=0.1),

        # OneOf([
        #     GridDistortion(p=1.0),
        #     ElasticTransform(p=1.0)
        # ], p=0.2),
        # OneOf([
        #     CLAHE(p=1.0),
        #     IAASharpen(p=1.0),
        # ], p=0.2)
    ], p=p)


def strong_aug2(p=1.0):
    return Compose([
        Flip(p=0.75),  # ok
        # RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=0.2, p=1.0, brightness_by_max=False),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, border_mode=cv2.BORDER_CONSTANT, p=0.2),

        OneOf([
            IAASharpen(p=1),
            Blur(blur_limit=5, p=1.0),
            MedianBlur(blur_limit=5, p=1.0),
            MotionBlur(p=1.0),
        ], p=0.6),

        OneOf([
            HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=1.0),
            RGBShift(p=1.0),
            RandomGamma(p=1),
        ], p=0.3),

        IAAAdditiveGaussianNoise(p=.2),
    ], p=p)


class Albu():
    def __call__(self, image, mask):
        augmentation = strong_aug() #just_brightness_flip() #strong_aug()

        data = {"image": image, "mask": mask}
        augmented = augmentation(**data)

        image, mask = augmented["image"], augmented["mask"]
        return image, mask


class Albu_test():
    def __call__(self, image, mask):
        augmentation = Compose([
                            # Flip(p=0.75),
                            # RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0)
                            # ShiftScaleRotate(p=0.2, shift_limit=0.1, scale_limit=0.1, rotate_limit=45, border_mode=cv2.BORDER_CONSTANT),
                            # Blur(blur_limit=5, p=0.5),
                            # MedianBlur(blur_limit=5, p=0.5),
                            # MotionBlur(p=0.5),
                            # GaussNoise(p=0.5),
                            # HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=1.0),
                            RGBShift(p=1.0)
                            # GridDistortion(p=1.0),
                            # ElasticTransform(p=1.0)
                            # CLAHE(p=0.5),
                            # IAASharpen(p=0.5)

                        ], p=1.0)

        data = {"image": image, "mask": mask}
        augmented = augmentation(**data)

        image, mask = augmented["image"], augmented["mask"]

        return image, mask



class aug_lab():
    def __call__(self, image, mask):
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # img_slice = image[:, :, 0]

        augmentation = Compose([
            RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=0.2, p=1.0,brightness_by_max=False),
        ], p=1.0)

        data = {"image": image, "mask": mask}
        augmented = augmentation(**data)

        image, mask = augmented["image"], augmented["mask"]

        # image[:, :, 0] = img_slice
        # image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

        return image, mask



if __name__ == "__main__":
    # aug = Albu_test()
    # aug = Albu()

    aug = aug_lab()

    # img = cv2.imread('../data/images/00000.png', 1)
    mask = cv2.imread('../data/masks/00000.png', 0)

    for i in range(100):
        img = cv2.imread('../data/images/00000.png', 1)
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
        # img_slice = img[:,:,0]
        # img_slice, out_mask = aug(img_slice, mask)
        # img[:,:,0] = img_slice
        # img = cv2.cvtColor(img,cv2.COLOR_LAB2BGR)
        # out_img = img
        # cv2.imshow('img', out_img)

        img = cv2.imread('../data/images/00000.png', 1)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        # img_slice = img[:, :, 0]
        out_img, out_mask = aug(img, mask)
        cv2.imshow('img', out_img)
        cv2.imshow('mask', out_mask)

        cv2.waitKey()
