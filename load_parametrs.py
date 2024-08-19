# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 18:09:16 2024

@author: EVM
"""
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def build_model(num_classes):
    # загрузка модели предобученную на датасете coco
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    
    # Размер входа предикторной части
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Заменяем последние слои детекции с учётом количества наших классов и загруженной модели
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Размер входа сегментной части
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # Заменяем последние слои сегментации
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model