import argparse
import os
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
from models import API_Net 
from datasets import RandomDataset, BatchDataset, BalancedBatchSampler
from utils import accuracy, AverageMeter, save_checkpoint
import tensorboardX


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=10, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
# parser.add_argument('--evaluate-freq', default=10, type=int,
#                     help='the evaluation frequence')
parser.add_argument('--resume', default='./checkpoint.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
#                     help='evaluate model on validation set')
# parser.add_argument('--pretrained', dest='pretrained', action='store_true',
#                     help='use pre-trained model')
parser.add_argument('--n_classes', default=2, type=int,
                    help='the number of classes')
parser.add_argument('--n_classes_total', default=5, type=int,
                    help='the overall number of classes')
parser.add_argument('--n_samples', default=5, type=int,
                    help='the number of samples per class')
parser.add_argument('--train_list', default='data_list/trycode.txt', type=str,
                    help='path to tensorboard')
parser.add_argument('--val_list', default='data_list/trycode.txt', type=str,
                    help='path to tensorboard')
parser.add_argument('--tensorboard_path', default='tensorboard_logs', type=str,
                    help='path to tensorboard')
parser.add_argument('--model_output_path', default='model_save', type=str,
                    help='path to save models')
parser.add_argument('--model_name', default='forgettowritename', type=str,
                    help='name of the model when saving')


best_prec1 = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    global args, best_prec1
    args = parser.parse_args()
    torch.manual_seed(2)
    torch.cuda.manual_seed_all(2)
    np.random.seed(2)

    n_classes_total = args.n_classes_total

    # create model
    model = API_Net(num_classes=args.n_classes_total)
    model = model.to(device)
    model.conv = nn.DataParallel(model.conv)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer_conv = torch.optim.SGD(model.conv.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    fc_parameters = [value for name, value in model.named_parameters() if 'conv' not in name]
    optimizer_fc = torch.optim.SGD(fc_parameters, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.resume:
        if os.path.isfile(args.resume):
            print('loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer_conv.load_state_dict(checkpoint['optimizer_conv'])
            optimizer_fc.load_state_dict(checkpoint['optimizer_fc'])
            print('loaded checkpoint {}(epoch {})'.format(args.resume, checkpoint['epoch']))
        else:
            print('no checkpoint found at {}'.format(args.resume))


    cudnn.benchmark = True
    # Data loading code
    train_list = args.train_list
    train_dataset = BatchDataset(train_list=train_list,
                                 transform=transforms.Compose([
                                     transforms.Resize([512,512]),
                                     transforms.RandomCrop([448,448]),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225)
                                     )]))
                                            
    train_sampler = BalancedBatchSampler(train_dataset, args.n_classes, args.n_samples)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=train_sampler,
        num_workers=args.workers, pin_memory=True)
    scheduler_conv = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_conv, 100*len(train_loader))
    scheduler_fc = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_fc, 100*len(train_loader))

    val_list = args.val_list
    val_dataset = RandomDataset(val_list=val_list,
                                transform=transforms.Compose([
                                    transforms.Resize([512, 512]),
                                    transforms.CenterCrop([448, 448]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        mean=(0.485, 0.456, 0.406),
                                        std=(0.229, 0.224, 0.225)
                                    )]))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    step = 0
    tensorboard_path = args.tensorboard_path
    model_output_path = args.model_output_path
    model_name = args.model_name

    print('START TIME:', time.asctime(time.localtime(time.time())))
    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, model, criterion, optimizer_conv, scheduler_conv, optimizer_fc, scheduler_fc, epoch, step,
              tensorboard_path, model_output_path, model_name, val_loader, n_classes_total)


def train(train_loader, model, criterion, optimizer_conv, scheduler_conv, optimizer_fc, scheduler_fc, epoch, step,
          tensorboard_path, model_output_path, model_name, val_loader, n_classes_total):
    global best_prec1

    batch_time = AverageMeter()
    data_time = AverageMeter()
    softmax_losses = AverageMeter()
    rank_losses = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # top5 = AverageMeter()
    train_writer = tensorboardX.SummaryWriter(tensorboard_path)

    # switch to train mode
    end = time.time()
    rank_criterion = nn.MarginRankingLoss(margin=0.05)
    softmax_layer = nn.Softmax(dim=1).to(device)

    for i, (input, target) in enumerate(train_loader):
        model.train()

        # measure data loading time
        data_time.update(time.time() - end)
        input_var = input.to(device)
        target_var = target.to(device).squeeze()


        # compute output
        logit1_self, logit1_other, logit2_self, logit2_other, labels1, labels2 = model(input_var, target_var, flag='train')
        batch_size = logit1_self.shape[0]
        labels1 = labels1.to(device)
        labels2 = labels2.to(device)

        self_logits = torch.zeros(2*batch_size, n_classes_total).to(device)
        other_logits= torch.zeros(2*batch_size, n_classes_total).to(device)
        self_logits[:batch_size] = logit1_self
        self_logits[batch_size:] = logit2_self
        other_logits[:batch_size] = logit1_other
        other_logits[batch_size:] = logit2_other
        # print(f'logit1_self {logit1_self}, logit2_self {logit2_self}')
        # print(f'labels1 {labels1}, labels2 {labels2}')

        # compute loss
        logits = torch.cat([self_logits, other_logits], dim=0)
        targets = torch.cat([labels1, labels2, labels1, labels2], dim=0)
        # print(f'train logits, targets: {logits}, {targets}')
        softmax_loss = criterion(logits, targets)

        self_scores = softmax_layer(self_logits)[torch.arange(2*batch_size).to(device).long(),
                                                         torch.cat([labels1, labels2], dim=0)]
        other_scores = softmax_layer(other_logits)[torch.arange(2*batch_size).to(device).long(),
                                                         torch.cat([labels1, labels2], dim=0)]
        flag = torch.ones([2*batch_size, ]).to(device)
        rank_loss = rank_criterion(self_scores, other_scores, flag)

        loss = softmax_loss + rank_loss

        # measure accuracy and record loss
        prec1 = accuracy(logits, targets, 1)
        # prec5 = accuracy(logits, targets, 5)
        losses.update(loss.item(), 2*batch_size)
        softmax_losses.update(softmax_loss.item(), 4*batch_size)
        rank_losses.update(rank_loss.item(), 2*batch_size)
        top1.update(prec1, 4*batch_size)
        # top5.update(prec5, 4*batch_size)

        # compute gradient and do SGD step
        optimizer_conv.zero_grad()
        optimizer_fc.zero_grad()
        loss.backward()

        # if epoch >= 8:
        optimizer_conv.step()
        scheduler_conv.step()
        optimizer_fc.step()
        scheduler_fc.step()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Train results: \t Time: {time}\nStep: {step}\t Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'SoftmaxLoss {softmax_loss.val:.4f} ({softmax_loss.avg:.4f})\t'
                  'RankLoss {rank_loss.val:.4f} ({rank_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, softmax_loss=softmax_losses, rank_loss=rank_losses,
                   top1=top1, step=step, time=time.asctime(time.localtime(time.time()))))

        # write in tensorboard
        train_writer.add_scalar('train_loss', losses.avg, epoch)
        train_writer.add_scalar('train_top1', top1.avg, epoch)
        train_writer.add_scalar('learning_rate_conv', optimizer_conv.param_groups[0]['lr'], epoch)
        train_writer.add_scalar('learning_rate_fc', optimizer_fc.param_groups[0]['lr'], epoch)

        if i == len(train_loader) - 1:
            print(f'before validate {torch.cuda.memory_allocated()}')
            prec1_val, loss_val = validate(val_loader, model, criterion)
            print(f'after validate {torch.cuda.memory_allocated()}')

            # remember best prec@1 and save checkpoint
            if not os.path.exists(model_output_path):
                os.makedirs(model_output_path)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint(save_path=model_output_path, state={
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer_conv': optimizer_conv.state_dict(),
                'optimizer_fc': optimizer_fc.state_dict(),
            }, is_best=is_best, saved_file=os.path.join(model_output_path, str(epoch) + '_' + model_name))

            train_writer.add_scalar('val_loss', loss_val, epoch)
            train_writer.add_scalar('val_top1', prec1_val, epoch)

        step = step + 1

    return step


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    softmax_losses = AverageMeter()
    top1 = AverageMeter()
    # top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):

            input_val = input.to(device)
            target_val = target.to(device).squeeze()

            # compute output
            logits_val = model(input_val, targets=None, flag='val')
            # print(f'train logits, targets_val: {logits_val}, {target_val}')

            if target_val.dim() != 0:
                # batch size cannot be 1
                softmax_loss = criterion(logits_val, target_val)

                prec1= accuracy(logits_val, target_val, 1)
                # prec5 = accuracy(logits, target_var, 5)
                softmax_losses.update(softmax_loss.item(), logits_val.size(0))
                top1.update(prec1, logits_val.size(0))
                # top5.update(prec5, logits.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    print('Validation results: \t Time: {time}\nTest: [{0}/{1}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'SoftmaxLoss {softmax_loss.val:.4f} ({softmax_loss.avg:.4f})\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                            i, len(val_loader), batch_time=batch_time, softmax_loss=softmax_losses,
                            top1=top1, time=time.asctime(time.localtime(time.time()))))
        print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg, softmax_losses.avg


if __name__ == '__main__':
    main()
