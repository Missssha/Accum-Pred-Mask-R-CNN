#Часть необходимая чтобы не создавалось несколько потоков и процессор не перегружался
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Импорт файлов для доп. функций
import transforms
import coco_utils
import coco_eval
from engine import train_one_epoch, evaluate
# from load_dataset import get_prediction, segment_instance
import utils
from PIL import Image
import cv2
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch
import random

import torchvision.transforms as T
import torchvision
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def get_coloured_mask(mask):
    """
    Функция возвращает маску случайного цвета для каждого объекта
    """
    colours = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [80, 70, 180],
               [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190]]
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask == 1] = colours[random.randrange(0, 10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask

def get_prediction(img_path, model):
    """
    Параметры:
        - img_path - путь к папке с проверяемыми изображениями
        - confidence - уверенность в предсказании от 0 до 1
    Метод:
        - Фотография извлекается из папки
        - Конвкертируется в тензор с помощью torch transforme
        - Подается в модель
        - Вычисляются маски, класс изображения, граничныее рамки

    """
    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)

    CLASS_NAMES = ['__background__', 'battery']
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    img = img.to(device)
    pred = model([img])
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    print(f'pred_score - {pred_score}')
    pred_t = [pred_score.index(x) for x in pred_score][-1]
    masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    pred_class = [CLASS_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    # masks = masks[:pred_t+1]
    # masks = masks[0]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return masks, pred_boxes, pred_class, pred_score[0]

def segment_instance(img_path, path_weight, filename, rect_th=2, text_size=2, text_th=2):
    """
    segment_instance
      Параметры:
        - img_path - путь к папке с проверяемыми изображениями
        - confidence- уверенность в предсказании от 0 до 1
        - rect_th - толшина линий прямоугольника
        - text_size - размер текста
        - text_th - толщина текста
      Метод:
        - Предсказание берется из get_prediction
        - Каждое изображение считывается при помощи opencv
        - Каждой маске дается случайный цвет функцией get_coloured_mask
        - each mask is added to the image in the ration 1:0.8 with opencv

    """
    model = torch.load(path_weight, map_location=torch.device('cpu'))
    model.eval()
    CLASS_NAMES = ['__background__', 'battery']
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    masks, boxes, pred_cls, pred_score = get_prediction(img_path, model)
    # print("masks.shape", masks.shape)
    # pred_score = round(pred_score, 2)
    # print("MASKA", masks.shape)
    print("PRED_SCORE", pred_score)
    confidence = pred_score
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # for i in range(len(masks)):
    # Проверка размерности маски, одно предсказание или несколько
    # Если несколько, то брать только первую маску, если одно - то маску без размерности
    list_shape = []
    for i in range(1):
      list_shape  = masks.shape
      if len(list_shape) > 2:
          res_mask = masks[i]
      else:
          res_mask = masks
      # rgb_mask = get_coloured_mask(res_mask)

      # img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
      # print("IMG", img.shape, img)
      boxes[i][0] = list(boxes[i][0])
      boxes[i][1] = list(boxes[i][1])
      for j in range(2):
          for u in range(2):
            boxes[i][j][u] = int(boxes[i][j][u])
      boxes[i][0] = tuple(boxes[i][0])
      boxes[i][1] = tuple(boxes[i][1])
      cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
      cv2.putText(img, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)

    # Расчет координат, если хотим выделить только изображение в предсказанной рамке
    koor1, koor2 = boxes[0]
    x1, y1  = koor1
    x2, y2 = koor2

    plt.figure(figsize=(10,15))
    # plt.title(f"Результат с точностью {confidence}")
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    result_dir = path_res
    plt.savefig(result_dir+filename, bbox_inches='tight', pad_inches=0)
    plt.show()

    return x1, y1, x2, y2, res_mask

######
# Библиотеки для подачи аргументов при запуске файла
from sys import argv
# script, path_weight, path_val, path_res = argv

# Закоментированные строки для определения путей из PyCharm, остальные для запуска из консоли
# model_cus = torch.load(path_weight, map_location=torch.device('cpu'))


def main(path_weight, path_val, path_res):
    model = torch.load(path_weight, map_location=torch.device('cpu'))
    model.eval()
    CLASS_NAMES = ['__background__', 'battery']
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    root = path_val
    img_path = 0
    imgs = list(sorted(os.listdir(root)))
    print("ДЛИНА СПИСКА", len(imgs))
    for i in range(0, len(imgs)):
        img_path = os.path.join(root, imgs[i])
        filename = f'result19_07{i}'
        # segment_instance(img_path, filename)
        x1, y1, x2, y2, mask_after = segment_instance(img_path, path_weight, filename)
        # print(x1, y1, x2, y2, mask_after)

    return x1, y1, x2, y2, mask_after

################
# if __name__ == "__main__":

script, path_weight, path_val, path_res = argv
# #
x1, y1, x2, y2, mask_after = main(path_weight, path_val, path_res)
#
# print("Mask after", mask_after.shape)
# # Дополнительная часть, чтобы посмотреть предсказанную маску (можно закоментить, чтобы часто не показывалось)
# rgb_mask = get_coloured_mask(mask_after)
#
# imgsq = sorted(os.listdir(path_val))
# imgs = os.path.join(path_val, imgsq[0])
#
# img = cv2.imread(imgs)[...,::-1]
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # img = cv2.imread('C:/Users/EVM/Documents/camera_test/val/seg/normal_with_blick.jpg')[...,::-1]
# # img = cv2.imread('C:/Users/EVM/Documents/camera_test/val/seg/3koda.jpg')[...,::-1]
# img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
# plt.imshow(img)
# plt.show()





