import argparse
import os
import time

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from src.squeezenet_torch import SqueezeNet


def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--last_epoch', type=int, default=0)
    parser.add_argument('--end_epoch', type=int, default=10)
    parser.add_argument('--device', type=int)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # cudnn related setting
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    print("cudnn related setting", flush=True)
    torch.cuda.set_device(args.device)

    model = SqueezeNet().cuda()
    print("create model", flush=True)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()
    print("create loss", flush=True)
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.01,
        momentum=0.9,
        weight_decay=1e-5,
        nesterov=True
    )
    print("create optimizer", flush=True)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        args.data_path,
        transforms.Compose([
            transforms.RandomResizedCrop(227),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )
    print("create imagenet", flush=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    print("create loader", flush=True)

    for epoch in range(args.last_epoch, args.end_epoch):
        # print(f"{epoch} epoch begin", flush=True)
        # train for one epoch
        batch_time = AverageMeter()
        losses = AverageMeter()

        # switch to train mode
        model.train()

        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            # print(f"{i} step begin", flush=True)
            # compute output
            input = input.cuda(non_blocking=True)
            output = model(input)
            target = target.cuda(non_blocking=True)
            loss = criterion(output, target)
            # compute gradient and do update step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # measure accuracy and record loss
            # loss value for each batch
            losses.update(loss.item(), input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # print(i, flush=True)
        print("epoch: {:3d}, epoch time: {:5.3f}, steps: {:5d}, per step time: {:5.3f}, avg loss: {:5.3f}".format(
            epoch+1, batch_time.sum, batch_time.count, batch_time.avg, losses.avg
        ), flush=True)
        #torch.save(model.state_dict(), os.path.join(args.train_path, f"efficientnetb1-{epoch+1}.pth"))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
