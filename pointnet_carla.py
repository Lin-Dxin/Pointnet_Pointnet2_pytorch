from data_utils.CarlaDataLoader import CarlaDataset
from torch.utils.data import DataLoader
import numpy as np
import time
from models.pointnet_sem_diy import get_model, get_loss
import torch
from tqdm import tqdm


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
    train_dataset = CarlaDataset(split='train')
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0,
                              pin_memory=True, drop_last=True)
    test_dataset = CarlaDataset(split='test')
    test_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0,
                             pin_memory=True, drop_last=True)
    # print(train_dataset.__len__())
    # print(test_dataset.__len__())
    numclass = 24
    classifier = get_model(numclass).cuda()
    criterion = get_loss().cuda()
    classifier.apply(inplace_relu)
    learning_rate = 0.001
    decay_rate = 0.0001
    step_size = 10
    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = step_size
    temp = CarlaDataset.label_weights
    weights = torch.Tensor(temp).cuda()

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
        lr = max(learning_rate * (decay_rate ** (epoch // step_size)), LEARNING_RATE_CLIP)
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
            points, target = points.float().cuda(), target.long().cuda()
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
            # total_seen += (BATCH_SIZE * NUM_POINT)
            loss_sum += loss
        print('Training mean loss: %f' % (loss_sum / num_batches))
        print('Training accuracy: %f' % (total_correct / float(total_seen)))

        with torch.no_grad():
            num_batches = len(test_loader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            labelweights = np.zeros(24)
            total_seen_class = [0 for _ in range(24)]
            total_correct_class = [0 for _ in range(24)]
            total_iou_deno_class = [0 for _ in range(24)]
            classifier = classifier.eval()
            print('---- EPOCH %03d EVALUATION ----' % (epoch + 1))
            for i, (points, target) in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
                points = points.data.numpy()
                points = torch.Tensor(points)
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)

                seg_pred, trans_feat = classifier(points)
                pred_val = seg_pred.contiguous().cpu().data.numpy()
                seg_pred = seg_pred.contiguous().view(-1, 24)
                batch_label = target.cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                loss = criterion(seg_pred, target, trans_feat, weights)
                loss_sum += loss
                pred_val = np.argmax(pred_val, 2)
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                # total_seen += (BATCH_SIZE * NUM_POINT)
                tmp, _ = np.histogram(batch_label, range(25))
                labelweights += tmp

                for l in range(24):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))
        labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
        mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6))
        print('\neval mean loss: %f' % (loss_sum / float(num_batches)))
        print('\neval point avg class IoU: %f' % (mIoU))
        print('\neval point accuracy: %f' % (total_correct / float(total_seen)))
        print('\neval point avg class acc: %f' % (
            np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))
