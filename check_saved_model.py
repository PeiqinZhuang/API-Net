import torch

path = 'model_save/effb5/2546262/model_best.pth.tar'
checkpoint = torch.load(path)
print('loaded checkpoint {}(epoch {})'.format(path, checkpoint['epoch']))
