from data_utils.CarlaDataLoader_npy import CarlaDataset
from torch.utils.data import DataLoader
import numpy as np
import time
from models.pointnet_semseg_carla import get_model, get_loss
import torch
from tqdm import tqdm
import datetime
import os
import sys
import logging
from pathlib import Path


classes = ['Unlabeled', 'Building', 'Fence', 'Other', 'Pedestrian', 'Pole', 'RoadLine', 'Road',
           'SideWalk', 'Vegetation', 'Vehicles', 'Wall', 'TrafficSign', 'Sky', 'Ground', 'Bridge'
    , 'RailTrack', 'GuardRail', 'TrafficLight', 'Static', 'Dynamic', 'Water', 'Terrain']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)


if __name__ == '__main__':
    NEED_SPEED = False
    PROPOTION = [0.7, 0.2, 0.1]
    # prepare for log file
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath(timestr)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if NEED_SPEED:
        file_handler = logging.FileHandler('%s/logs/4d_train.txt' % experiment_dir)
    else:
        file_handler = logging.FileHandler('%s/logs/3d_train.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


    def log_string(str):
        logger.info(str)
        print(str)


    # train
    # config dataloader

    train_dataset = CarlaDataset(split='train', need_speed=NEED_SPEED, proportion=PROPOTION)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0,
                              pin_memory=True, drop_last=True)
    test_dataset = CarlaDataset(split='test', need_speed=NEED_SPEED, proportion=PROPOTION)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=0,
                             pin_memory=True, drop_last=True)
    # print(train_dataset.__len__())
    # print(test_dataset.__len__())
    log_string("The number of training data is: %d" % len(train_dataset))
    log_string("The number of test data is: %d" % len(test_dataset))

    numclass = 23
    classifier = get_model(numclass, need_speed=NEED_SPEED).to(device) # loading model
    criterion = get_loss().to(device) # loss function
    classifier.apply(inplace_relu)
    learning_rate = 0.001
    decay_rate = 0.0001
    step_size = 10
    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = step_size
    temp = CarlaDataset.label_weights
    weights = torch.Tensor(temp).to(device)

    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=decay_rate
    )


    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum


    epoch_num = 32
    best_iou = 0
    for epoch in range(epoch_num):
        log_string('**** Epoch %d  ****' % (epoch + 1))
        start_time = time.time()
        lr = max(learning_rate * (decay_rate ** (epoch // step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(train_loader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        classifier = classifier.train()

        for i, (points, target) in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
            # print(points.shape)
            # print(target)
            optimizer.zero_grad()
            points = points.data.numpy()
            points = torch.Tensor(points)
            points, target = points.float().to(device), target.long().to(device)
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, numclass)

            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, target, trans_feat, weights)
            loss.backward()
            optimizer.step()

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            # total_seen += (16 * 4096)
            total_seen += 16 * train_dataset.numpoints
            loss_sum += loss
            # break
        log_string('Training mean loss: %f' % (loss_sum / num_batches))
        log_string('Training accuracy: %f' % (total_correct / float(total_seen)))

        with torch.no_grad():
            num_batches = len(test_loader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            labelweights = np.zeros(23)
            total_seen_class = [0 for _ in range(23)]
            total_correct_class = [0 for _ in range(23)]
            total_iou_deno_class = [0 for _ in range(23)]
            classifier = classifier.eval()
            print('---- EPOCH %03d EVALUATION ----' % (epoch + 1))
            for i, (points, target) in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
                points = points.data.numpy()
                points = torch.Tensor(points)
                points, target = points.float().to(device), target.long().to(device)
                points = points.transpose(2, 1)

                seg_pred, trans_feat = classifier(points)
                pred_val = seg_pred.contiguous().cpu().data.numpy()
                seg_pred = seg_pred.contiguous().view(-1, 23)
                batch_label = target.cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                loss = criterion(seg_pred, target, trans_feat, weights)
                loss_sum += loss
                pred_val = np.argmax(pred_val, 2)
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += 16 * train_dataset.numpoints
                # print(np.histogram(batch_label, range(24)))
                tmp, _ = np.histogram(batch_label, range(24))
                labelweights += tmp

                for l in range(0, 23):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))
                # break
        labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
        mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=float) + 1e-6))
        # sum = 0
        # valid = 0
        # for l in range(len(total_correct_class)):
        #     if(total_iou_deno_class != 0):
        #         valid = valid + 1
        #         sum += np.array(total_correct_class[l]) / (np.array(total_iou_deno_class[l], dtype=float) + 1e-6)
        # mIoU = sum / valid
        log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
        log_string('eval point avg class IoU: %f' % mIoU)
        log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))
        # log_string('eval point avg class acc: %f' % (
        #     np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=float) + 1e-6))))
        iou_per_class_str = '------- IoU --------\n'
        for l in range(numclass):
            iou_per_class_str += 'class %s weight: %.3f' % (
                seg_label_to_cat[l] + ' ' * (23 - len(seg_label_to_cat[l])), labelweights[l])
            if total_iou_deno_class[l] != 0:
                iou_per_class_str += ', IoU: %.3f \n' % (total_correct_class[l] / float(total_iou_deno_class[l]))
            else:
                iou_per_class_str += ', IoU: UnValid\n'
        log_string(iou_per_class_str)
        end_time = time.time()
        spd_time = end_time - start_time
        log_string('Spending Time: %f' % spd_time)
        log_string('Eval mean loss: %f' % (loss_sum / num_batches))
        log_string('Eval accuracy: %f' % (total_correct / float(total_seen)))
        if mIoU >= best_iou:
            best_iou = mIoU
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'class_avg_iou': mIoU,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')
        log_string('Best mIoU: %f' % best_iou)
