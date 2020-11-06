import os
import math

import argparse
import nsml
import torch
import torch.nn as nn
import torchvision.models as models

from data_loader import feed_infer
from data_local_loader import data_loader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from nsml import DATASET_PATH, IS_ON_NSML
from evaluation import evaluation_metrics

if IS_ON_NSML:
    TRAIN_DATASET_PATH = os.path.join(DATASET_PATH, 'train', 'train_data')
else:
    DATASET_PATH = '/home/dataset/keris/'


class ClsResNet(models.ResNet):
    """Model definition.

    You can use any model for the challenge. Feel free to modify this class.
    """

    def forward(self, x, extract=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x


def _infer(model, root_path, loader=None):
    """Local inference function for NSML infer.

    Args:
        model: instance. Any model is available.
        root_path: string. Automatically set by NSML.
        test_loader: instance. Data loader is defined in `data_local_loader.py`.

    Returns:
        predictions_str: list of string.
                         ['img_1,1,0,1,0,1,0,0,0', 'img_2,0,1,0,0,1,0,0,0', ...]
    """
    model.eval()

    if loader is None:
        loader = data_loader(root=os.path.join(root_path))

    list_of_fids = []
    list_of_preds = []

    for idx, (image, fid, _) in enumerate(loader):
        image = image.cuda()
        fc = model(image, extract=True)
        fc = fc.detach().cpu().numpy()
        fc = 1 * (fc > 0.5)

        list_of_fids.extend(fid)
        list_of_preds.extend(fc)

    predictions_str = []
    for idx, fid in enumerate(list_of_fids):
        test_str = fid
        for pred in list_of_preds[idx]:
            test_str += ',{}'.format(pred)
        predictions_str.append(test_str)

    return predictions_str


def bind_nsml(model):
    """NSML binding function.

    This function is used for internal process in NSML.
    Please modify this module according to your framework.
    """

    def save(dir_name, *args, **kwargs):
        os.makedirs(dir_name, exist_ok=True)
        state = {
            'model': model.state_dict(),
        }
        torch.save(state, os.path.join(dir_name, 'model.pth'))
        print('saved')

    def load(dir_name, *args, **kwargs):
        state = torch.load(os.path.join(dir_name, 'model.pth'))
        model.load_state_dict(state['model'])
        print('loaded')

    def infer(root_path, top_k=1):
        return _infer(model, root_path)

    nsml.bind(save=save, load=load, infer=infer)


def load_weight(model):
    """Weight loading function.

    You should put your weight file on root directory. The name of weight file
    should be 'checkpoint.pth'. If there is no 'checkpoint.pth' on root directory,
    the weights will be randomly initialized.
    """
    if os.path.isfile('checkpoint.pth'):
        state_dict = torch.load('checkpoint.pth')['state_dict']
        model.load_state_dict(state_dict, strict=True)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def local_eval(model, loader, gt_path):
    """Local debugging function.

    You can use this function for debugging. You may need dummy gt file.

    Args:
        model: instance.
        test_loader: instance.
        gt_path: string.

    Returns:
        metric_result: float. Performance of your method.
    """
    pred_path = 'pred.txt'
    feed_infer(pred_path, lambda root_path: _infer(model=model,
                                                   root_path=root_path,
                                                   loader=loader))
    metric_result = evaluation_metrics(pred_path, gt_path)
    return metric_result


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--num_classes", type=int, default=8)

    # Arguments for train mode
    args.add_argument("--num_epochs", type=int, default=50)
    args.add_argument("--base_lr", type=float, default=0.001)
    args.add_argument("--step_size", type=int, default=20)

    # These three arguments are reserved for nsml. Do not change.
    args.add_argument("--mode", type=str, default="train")
    args.add_argument("--iteration", type=str, default='0')
    args.add_argument("--pause", type=int, default=0)

    config = args.parse_args()

    model = ClsResNet(block=models.resnet.BasicBlock,
                      layers=[2, 2, 2, 2],
                      num_classes=config.num_classes)
    load_weight(model)
    criterion = nn.BCEWithLogitsLoss()

    model = model.cuda()
    criterion = criterion.cuda()

    optimizer = Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=config.base_lr, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

    if IS_ON_NSML:
        # This NSML block is mandatory. Do not change.
        bind_nsml(model)
        nsml.save('checkpoint')
        if config.pause:
            nsml.paused(scope=locals())

    if config.mode == 'train':
        # Local debugging block. This module is not mandatory.
        # But this would be quite useful for troubleshooting.
        train_loader = data_loader(root=DATASET_PATH, split='train')
        val_loader = data_loader(root=DATASET_PATH, split='val')
        num_batches = len(train_loader)

        for epoch in range(config.num_epochs):
            scheduler.step()
            model.train()

            total_loss = 0.0
            num_images = 0

            for iter_, (image, image_id, label) in enumerate(train_loader):
                image = image.cuda()
                label = label.cuda()

                pred = model(image)
                loss = criterion(pred, label)

                total_loss += loss.item() * image.size(0)
                num_images += image.size(0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_average = total_loss / float(num_images)

            if IS_ON_NSML:
                nsml.save(str(epoch + 1))

            gt_label = os.path.join(DATASET_PATH, 'train/train_data/val_label')
            acc = local_eval(model, val_loader, gt_label)
            print(f'[{epoch + 1}/{config.num_epochs}] '
                  f'Validation performance: {acc:.3f}')
            nsml.report(step=epoch, val_acc=acc)
            nsml.report(step=epoch, train_loss=loss_average)