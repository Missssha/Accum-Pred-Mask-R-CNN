# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:38:26 2024

@author: EVM
"""

from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import random

import torch
import torch.utils.data
import torchvision
#import torchvision.transforms as T

from engine import train_one_epoch, evaluate
import utils
import transforms as T
from torchvision.transforms import v2
from torchvision import transforms


#Бинаризация без цикла
def binarize(img):
    
  img=img.convert('L')
  img = np.array(img)
  img[img <128] = 0
  img[img >= 128] = 1
  img = Image.fromarray(img)
  
  return img

from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.utils.data
from PIL import Image

from vgg_to_mask import vgg2dict, dict2mask

import albumentations as A
from albumentations.pytorch import ToTensorV2

#Класс датасета
#Создает датасет из 2-х папок, одна - изображение, вторая - маска
#При вызове класса для создания датасета пользователь указывает путь к папке с изображениями и масками
#Ширину и высоту сжатого изображения
#Название папок для загрузки
#Пример использования
#dataset_0 = BatteryDataset('C:/Users/EVM/Documents/camera_test', 120, 160, 'image2', 'mask2')

from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.utils.data
from PIL import Image

from vgg_to_mask import vgg2dict, dict2mask, dict2binarymask

# Загрузка маски в виде анотации json
class Dataset_load(torch.utils.data.Dataset):
    def __init__(self, root, vgg_json, width, height, W_res, H_res, transforms=None):
        self.root = Path(root)
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.images: List[Path] = sorted(list(self.root.glob('*')))
        self.masks = vgg2dict(vgg_json)
        self.size = {'width': width, 'height': height}
        self.W_res = W_res
        self.H_res = H_res

    def __getitem__(self, idx):
        # load images ad masks
        
        img = Image.open(self.images[idx]).convert("RGB")
        img = img.resize((self.W_res, self.H_res), Image.Resampling.NEAREST)
        # print(img.size)
        mask = dict2mask(image_name=self.images[idx].name,
                         mask_dict=self.masks,
                         mask_width=self.size['width'],
                         mask_height=self.size['height'])
        
        binary_mask = dict2binarymask(image_name=self.images[idx].name,
                         mask_dict=self.masks,
                         mask_width=self.size['width'],
                         mask_height=self.size['height'])
        # print("MASK", type(mask))
        mask2 = Image.fromarray(mask).resize((self.W_res, self.H_res), Image.Resampling.NEAREST)
        binary_mask2 = Image.fromarray(binary_mask).resize((self.W_res, self.H_res), Image.Resampling.NEAREST)
        # print("MASK2", type(mask2), mask2.size)
        mask2 = np.array(mask2)
        binary_mask2 = np.array(binary_mask2)
        # print("AFTER", type(mask2), mask2.shape)
        mask=mask2
        binary_mask=binary_mask2
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background

        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        obj_ids_of_binary_mask = np.unique(binary_mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        obj_ids_of_binary_mask = obj_ids_of_binary_mask[1:]
        # print(obj_ids)

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]
        binary_masks = binary_mask == obj_ids_of_binary_mask[:, None, None]

        # get bounding box coordinates for each mask
        # num_objs = count_polygons
        # print("Количество объектов:" + str(len(obj_ids)))
        
        boxes = []
        for i in range(len(obj_ids)):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
            # print(boxes)
            # print(masks[i])
            # print("Количество объектов:" + str(num_objs))
        # print(masks)
        # print(boxes)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((len(obj_ids),), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        binary_masks = torch.as_tensor(binary_masks, dtype=torch.uint8)
        

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.ones((len(obj_ids),), dtype=torch.int64)
        # print("Количество объектов:" + str(len(obj_ids)))

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)
            # transformed = self.transforms(image=img, mask=masks)
            # img = transformed['image']
            # masks = transformed['masks']

        return img, target

        # if self.transforms is not None:
        #     img, masks = self.transforms(img, masks)
        # return img, target

    def __len__(self):
        return len(self.images) 

class Dataset_load_2(torch.utils.data.Dataset):
    def __init__(self, root, vgg_json, width, height, W_res, H_res, transforms=None):
        self.root = Path(root)
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.images: List[Path] = sorted(list(self.root.glob('*')))
        self.masks = vgg2dict(vgg_json)
        self.size = {'width': width, 'height': height}
        self.W_res = W_res
        self.H_res = H_res

    def __getitem__(self, idx):
        # load images ad masks
        
        img = Image.open(self.images[idx]).convert("RGB")
        # print(img.size)
        img = img.resize((self.W_res, self.H_res), Image.Resampling.NEAREST)
        # print(img.size)
        mask = dict2mask(image_name=self.images[idx].name,
                         mask_dict=self.masks,
                         mask_width=self.size['width'],
                         mask_height=self.size['height'])
        # print("MASK", self.images[idx])
        mask2 = Image.fromarray(mask).resize((self.W_res, self.H_res), Image.Resampling.NEAREST)
        # print("MASK2", type(mask2), mask2.size)
        mask2 = np.array(mask2)
        # print("AFTER", type(mask2), mask2.shape)
        mask=mask2
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background

        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.ones((num_objs,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms is not None:
            img, target = self.transforms(image = img, mask = target)
            # transformed = self.transforms(image=img, mask=masks)
            # img = transformed['image']
            # masks = transformed['masks']

        return img, target

        # if self.transforms is not None:
        #     img, masks = self.transforms(img, masks)
        # return img, target

    def __len__(self):
        return len(self.images)


# Загрузка маски в виде фоток
class BatteryDataset(torch.utils.data.Dataset):
    def __init__(self, root, W_in, H_in, image, mask, transforms=None):
        self.root = root
        self.transforms = transforms
        self.H = H_in
        self.W = W_in
        self.image = image
        self.mask = mask
        # Запись директрии изображения и маски
        self.imgs = list(sorted(os.listdir(os.path.join(self.root, self.image))))
        self.masks = list(sorted(os.listdir(os.path.join(self.root, self.mask))))

    def __getitem__(self, idx):
        # Объединение названий элементов папок
        img_path = os.path.join(self.root, self.image, self.imgs[idx])        
        mask_path = os.path.join(self.root, self.mask, self.masks[idx])
        
        img = Image.open(img_path).convert("RGB")
        img1 = img.convert('L')
        # Сжатие
        img = img.resize((self.W, self.H), Image.Resampling.NEAREST)
        
        mask = Image.open(mask_path)
        # Сжатие
        mask = mask.resize((self.W, self.H), Image.Resampling.NEAREST)
        mask = binarize(mask)

        mask = np.array(mask)
        
        obj_ids = np.unique(mask)
        # 1 - задний фон, удаляем его
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]

        # Получаем координаты границ боксов у каждой маски
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        # labels = torch.ones((num_objs,))
        # masks = torch.as_tensor(masks)

        image_id = torch.tensor([idx])
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.ones((num_objs,), dtype=torch.int64)
        # iscrowd = torch.zeros((num_objs,))

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img, target)
            # target['masks'] = self.transforms(target['masks'])
            target['masks'] = target['masks'].detach().numpy()
            target['masks'] = self.transforms(target['masks'], target)
            target['masks'] = torch.from_numpy(target['masks'])
            # transformed  = self.transforms(image=np.array(img), mask=mask)
            # image = transformed["image"]
            # mask = transformed["mask"]

        return img, target

    def __len__(self):
        return len(self.imgs)

#Транформация изображений для аугментации 
def get_transform_Batt(train):
    transforms = []
    # конвертация PIL в Torch
    transforms.append(T.ToTensor())
    if train:
        # Добавляем вращение по горизонтали
        # transforms.append(T.RandomHorizontalFlip(0.5))
        # Добавляем изменение преспективы
        # transforms.append(v2.RandomPerspective(distortion_scale = 0.5, p = 1.0))
        # Добавляем вращение
        # transforms.append(v2.RandomRotation(degrees=(0, 180)))
        # Вращение по вертикали
        # transforms.append(v2.RandomVerticalFlip(p=0.5))
        transforms.append(v2.ColorJitter(brightness=0.12, contrast=0.1, saturation=0.5, hue=0.4))
        transforms.append(v2.RandomEqualize())
        print(transforms)
        
    return T.Compose(transforms)

# Трансформация для json
def get_transform_json(train):
    return A.Compose(
        [
            A.Resize(160, 120),
            # A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
            # A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.9, contrast_limit=0.9, p=0.5),
            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], p=1
    )


def get_coloured_mask(mask):

    colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask == 1] = colours[random.randrange(0,10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask

def get_prediction(img_path, confidence, model):
    
    """
    get_prediction
      входые параметры:
        - img_path - путь к изображению/папке
        - confidence - порог уверенности
    """
    img = Image.open(img_path)
    print(img)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    img = img.to(device)
    pred = model([img])
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x>confidence][-1]
    # print((pred[0]['masks']>0.1).squeeze().detach().cpu().numpy())
    # print((pred[0]['masks']>0.2).squeeze().detach().cpu().numpy())
    masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    # print('MASKS > 50', masks)
    mask_one = masks.astype(int)
    # print('ONE', mask_one)
    pred_class = [CLASS_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    masks = masks[:pred_t+1]
    # print('MAX', np.max(masks))
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return masks, pred_boxes, pred_class

def segment_instance_1(img_path, filename, confidence=0.2, rect_th=2, text_size=2, text_th=2):
    """
    segment_instance
      parameters:
        - img_path - path to input image
        - confidence- confidence to keep the prediction or not
        - rect_th - rect thickness
        - text_size
        - text_th - text thickness
      method:
        - prediction is obtained by get_prediction
        - each mask is given random color
        - each mask is added to the image in the ration 1:0.8 with opencv
        - final output is displayed
    """
    masks, boxes, pred_cls = get_prediction(img_path, confidence, model)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(len(masks)):
      rgb_mask = get_coloured_mask(masks[i])
      img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
      boxes[i][0] = list(boxes[i][0])
      boxes[i][1] = list(boxes[i][1])
      for j in range(2):
          for u in range(2):
            boxes[i][j][u] = int(boxes[i][j][u])
      boxes[i][0] = tuple(boxes[i][0])
      boxes[i][1] = tuple(boxes[i][1])
      cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
      cv2.putText(img,pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    plt.figure(figsize=(20,30))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.show()