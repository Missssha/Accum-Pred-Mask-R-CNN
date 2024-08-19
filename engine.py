import math
import sys
import time
import torch

import torchvision.models.detection.mask_rcnn

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    loss_list = []
    model.train()
    # losses_of_epoch = {}
    
    metric_logger = utils.MetricLogger(delimiter="  ")    
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        # key = "loss_mask"
        # if key in loss_dict:
        #     del loss_dict[key]

        losses = sum(loss for loss in loss_dict.values())
    	
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        # print(losses_reduced)
        # print(loss_dict_reduced)
        # print(loss_dict_reduced.values())

        loss_value = losses_reduced.item()
    	# loss_list.append(loss_value)
        # print(loss_value)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        values = loss_dict_reduced.values()
        # list_of_loss = []
        # for value in values:
        #     list_of_loss.append(value.item())
        # print("loss_dict", loss_dict)
        split_meters = str(metric_logger).split(" ")
        losses_of_one_epoch = [[],[],[],[],[],[]]
        losses_of_one_epoch[0].append(float(split_meters[4]))
        losses_of_one_epoch[1].append(float(split_meters[8]))
        losses_of_one_epoch[2].append(float(split_meters[12]))
        losses_of_one_epoch[3].append(float(split_meters[16]))
        losses_of_one_epoch[4].append(float(split_meters[20]))
        losses_of_one_epoch[5].append(float(split_meters[24]))
        
    print('losses_of_one_epoch', losses_of_one_epoch)
    return losses_of_one_epoch
        # print('list_of_loss', list_of_loss)
        # print("losses_of_epoch", losses_of_epoch)




def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cpu.synchronize()
        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()
        
    # accumulate predictions from all images
    coco_evaluator.accumulate()
    result_str = coco_evaluator.summarize()
    
    # with open('C:\\Users\\EVM\\Documents\\camera_test\\metrics.txt', 'a') as file2:
    #     file2.write(str(metric_logger))
    #     file2.write('\n')
    #     file2.write(coco_eval_str)

    
    torch.set_num_threads(n_threads)
    return coco_evaluator, metric_logger

def train_model (model, lr, data_loader, device, num_epochs, print_freq, data_loader_test, result_box, result_seg):
    # Создаем список всех параметров (весов и смещений) модели, которые требуют градиентного обновления
    params = [p for p in model.parameters() if p.requires_grad]
    # momentum помогает ускорить обучение и сгладить траекторию обновления параметров, учитывая предыдущие градиенты.
    # optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9) #, weight_decay=0.0005)
    optimizer = torch.optim.Adam(params, lr=lr)
    
    # scheduler уменьшает lr каждые 3 эпохи на 10
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq)
        # Обновление lr
        lr_scheduler.step()
        print("EPOCH", epoch)
        # Проверка на тестовом датасете
        coco_evaluator, metric_logger = evaluate(model, data_loader_test, device)
        result_box.append(coco_evaluator.coco_eval['bbox'].stats)
        result_seg.append(coco_evaluator.coco_eval['segm'].stats)
        
    return optimizer, lr_scheduler, result_box, result_seg