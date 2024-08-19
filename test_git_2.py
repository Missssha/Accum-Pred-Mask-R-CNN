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
script, path_img, path_json = argv

Weight = 960 # ширина входного изображения
Height = 1280 # высота входного изображения
num_epoch = 10 # Кол-во эпох
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
random_num = random.randint(1, 85)
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

num_epochs = num_epoch
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

# Сохранение модели
data = 28
torch.save(model_10_28_11, f"model_{num_epochs}_{data}_17.pt")

#model_10_28_11 = torch.load(path_weight)
#model_10_28_11 = torch.load('C:/Users/EVM/Documents/camera_test/weights/model_7_27_.pt')

import torchvision.transforms as T
import torchvision
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def get_coloured_mask(mask):
    """
    random_colour_masks
      parameters:
        - image - predicted masks
      method:
        - the masks of each predicted object is given random colour for visualization
    """
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
      parameters:
        - img_path - path of the input image
        - confidence - threshold to keep the prediction or not
      method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - masks, classes and bounding boxes are obtained from the model and soft masks are made binary(0 or 1) on masks
          ie: eg. segment of cat is made 1 and rest of the image is made 0

    """
    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    print(img)

    img = img.to(device)
    pred = model([img])
    print(pred)
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x>confidence][-1]
    print((pred[0]['masks']>0.1).squeeze().detach().cpu().numpy())
    print((pred[0]['masks']>0.2).squeeze().detach().cpu().numpy())
    masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    #print(pred[0]['labels'].numpy().max())
    pred_class = [CLASS_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    masks = masks[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return masks, pred_boxes, pred_class

def segment_instance(img_path, filename, confidence=0.2, rect_th=2, text_size=2, text_th=2):
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
      # filename = f'img_{i}.png'
    plt.figure(figsize=(20,30))
    plt.title(f"Результат с точностью {confidence}")
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.show()


model_10_28_11.eval()
CLASS_NAMES = ['__background__', 'battery']
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_10_28_11.to(device)

# Вывод предсказанных масок для изображений из папок
root = 'C:/Users/EVM/Documents/camera_test/val/'
#root = path_val
image = 'val'
model = model_10_28_11
confidence=0.72
imgs = list(sorted(os.listdir(os.path.join(root, image))))
print("ДЛИНА СПИСКА", len(imgs))
for i in range(0, len(imgs)):
    img_path = os.path.join(root, image, imgs[i])
    print("IMG_PATH", img_path)
    filename = f'result28_06{i}'
    print("FILE_NAME", filename)
    segment_instance(img_path, filename, confidence=confidence)

print('All')