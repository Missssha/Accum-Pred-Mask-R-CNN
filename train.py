#Часть необходимая чтобы не создавалось несколько потоков и процессор не перегружался
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#Испорт свох файлов с функциями для датасета и создания модели
from load_parametrs import build_model
from load_dataset import BatteryDataset, get_transform_Batt, Dataset_load

# Импорт файлов для доп. функций
import transforms
import coco_utils
import coco_eval
from engine import train_one_epoch, evaluate
import utils

from PIL import Image
import cv2
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch
import random

# Библиотеки для подачи аргументов при запуске файла
from sys import argv
script, path_img, path_json, num_epoch, data_cus = argv

Weight = 960 # ширина входного изображения
Height = 1280 # высота входного изображения
# num_epoch = 10 # Кол-во эпох
lr = 0.001 # learning rate для оптимизатора  Anam
H_res = 160 # Высота после сжатия
W_res = 120 # Ширина после сжатия

# Загрузка датасета для обучения и тестового для оценивания
dataset_2 = Dataset_load(
    #root='C:/Users/EVM/Documents/camera_test/image2', #Папка с фотографиями
    root = path_img,
    #vgg_json='C:/Users/EVM/Documents/camera_test/battery_test/Anatation4.json', # Json файл с координатами масок
    vgg_json = path_json,
    height=Height,
    width=Weight,
    W_res = W_res,
    H_res = H_res,
    transforms = get_transform_Batt(train=True))

dataset_test_2 = Dataset_load(
    #root='C:/Users/EVM/Documents/camera_test/image2',
    root = path_img,
    #vgg_json='C:/Users/EVM/Documents/camera_test/battery_test/Anatation4.json',
    vgg_json = path_json,
    height=Height,
    width=Weight,
    W_res = W_res,
    H_res = H_res,
    transforms = get_transform_Batt(train=False))

torch.manual_seed(1)
# Случайным образом пересталяет наборы в датасете
indices = torch.randperm(len(dataset_2)).tolist()
# Деление на train и test
dataset_2 = torch.utils.data.Subset(dataset_2, indices[:-20])
dataset_test_2 = torch.utils.data.Subset(dataset_test_2, indices[-20:])

# Деление на бачи
data_loader_2 = torch.utils.data.DataLoader(
    dataset_2, batch_size=2, shuffle=True, num_workers=0,
    collate_fn=utils.collate_fn)

data_loader_test_2 = torch.utils.data.DataLoader(
    dataset_test_2, batch_size=1, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn)

# Вывод аугментированного датасета
random_num = random.randint(1, 66)
#random_num = 32
row = 1
colomns = 2
fig = plt.figure(figsize = (10,15))
fig.add_subplot(row, colomns, 1)
plt.imshow(np.transpose(dataset_2[random_num][0].cpu().numpy(), (1,2,0)), cmap = 'gray')
plt.title('Изображение')
fig.add_subplot(row, colomns, 2)

plt.imshow(np.transpose(dataset_2[random_num][1]['masks'].cpu().numpy(), (1,2,0)))
plt.title('Маска')

# plt.axis('off')  # Отключение осей
plt.show()

#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = coco_utils.torch.device('cpu')
print(device)
# Кол-во классов 1 - фон, 1 - класс
num_classes = 1+1

# Применяем ф-цию загрузки предобученной модели
model_10_28_11 = build_model(num_classes)
# Исполняем на устройстве
model_10_28_11.to(device)

# Создаем список всех параметров (весов и смещений) модели, которые требуют градиентного обновления
params = [p for p in model_10_28_11.parameters() if p.requires_grad]
# momentum помогает ускорить обучение и сгладить траекторию обновления параметров, учитывая предыдущие градиенты.
# optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9) #, weight_decay=0.0005)

# optimizer = torch.optim.SGD(params, lr=0.002,
                            # momentum=0.9, weight_decay=0.0001)

optimizer = coco_utils.torch.optim.Adam(params, lr=lr)

# scheduler уменьшает lr каждые 3 эпохи на 10
lr_scheduler = coco_utils.torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

num_epochs = int(num_epoch)
#Список списков, который хранит все параметры по точности сегментации по боксам
result_box_10 = [[] for _ in range(num_epochs)]
#Список списков, который хранит все параметры по точности сегментации по маскам
result_seg_10 = [[] for _ in range(num_epochs)]
for epoch in range(num_epochs):
    train_one_epoch(model_10_28_11, optimizer, data_loader_2, device, epoch, print_freq=10)
    lr_scheduler.step()
    coco_evaluator_10, metric_logger_10 = evaluate(model_10_28_11, data_loader_test_2, device)
    result_box_10.append(coco_evaluator_10.coco_eval['bbox'].stats)
    result_seg_10.append(coco_evaluator_10.coco_eval['segm'].stats)

# Убираем пустые первые элементы у боксов
result_box = [[] for _ in range(len(result_box_10))]
for i in range(0, 12):
    for j in range(7, 13):
        result_box[i].append(result_box_10[j][i])


# Убираем пустые первые элементы у масок
result_seg = [[] for _ in range(len(result_seg_10))]
for i in range(0, 12):
    for j in range(7, 13):
        result_seg[i].append(result_seg_10[j][i])

# Сохранение модели
data = int(data_cus)
torch.save(model_10_28_11, f"model_{num_epochs}_{data}_17.pt")

label_x = np.linspace(1, num_epochs-1, num_epochs-1)

plt.figure(figsize=(15, 5))
plt.subplot(131)
# plt.bar(names, values)
plt.plot(label_x, result_box[0])
plt.title('Средняя точность (все точки с шагом 0.05) ')

plt.subplot(132)
plt.plot(label_x, result_box[2])
plt.title('Средняя точность (все точки с порогом 0.75) ')

plt.subplot(133)
plt.plot(label_x, result_box[6])
plt.title('Средний отзыв по всем точкам')

plt.suptitle(f'Результат по боксам после {num_epoch} эпох')
plt.show()

plt.figure(figsize=(15, 5))
plt.subplot(131)
# plt.bar(names, values)
plt.plot(label_x, result_seg[0])
plt.title('Средняя точность (все точки с шагом 0.05) ')

plt.subplot(132)
plt.plot(label_x, result_seg[2])
plt.title('Средняя точность (все точки с порогом 0.75) ')

plt.subplot(133)
plt.plot(label_x, result_seg[6])
plt.title('Средний отзыв по всем точкам')

plt.suptitle(f'Результат по маска после {num_epoch} эпох')
plt.show()

