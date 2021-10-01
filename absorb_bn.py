import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
#from torchviz import make_dot
from matplotlib import pyplot as plt
import pdb
import sys
import datetime
import os
from vgg_bn import *
from vgg_bn_bias import *
from resnet import *
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ANN to be later converted to SNN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu',                    default=True,               type=bool,      help='use gpu')
    parser.add_argument('--log',                    action='store_true',                        help='to print the output on terminal or to log file')
    parser.add_argument('-s','--seed',              default=0,                  type=int,       help='seed for random number')
    parser.add_argument('--dataset',                default='CIFAR10',          type=str,       help='dataset name', choices=['MNIST','CIFAR10','CIFAR100'])
    parser.add_argument('--batch_size',             default=64,                 type=int,       help='minibatch size')
    parser.add_argument('-a','--architecture',      default='VGG16',            type=str,       help='network architecture', choices=['VGG4','VGG5','VGG9','VGG11','VGG13','VGG16','VGG19','RESNET12','RESNET20','RESNET34'])
    parser.add_argument('-lr','--learning_rate',    default=1e-2,               type=float,     help='initial learning_rate')
    parser.add_argument('--pretrained_ann',         default='',                 type=str,       help='pretrained model to initialize ANN')
    parser.add_argument('--test_only',              action='store_true',                        help='perform only inference')
    parser.add_argument('--epochs',                 default=500,                type=int,       help='number of training epochs')
    parser.add_argument('--lr_interval',            default='0.45 0.70 0.90',   type=str,       help='intervals at which to reduce lr, expressed as %%age of total epochs')
    parser.add_argument('--lr_reduce',              default=5,                 type=int,       help='reduction factor for learning rate')
    parser.add_argument('--optimizer',              default='SGD',             type=str,        help='optimizer for SNN backpropagation', choices=['SGD', 'Adam'])
    parser.add_argument('--dropout',                default=0.2,                type=float,     help='dropout percentage for conv layers')
    parser.add_argument('--kernel_size',            default=3,                  type=int,       help='filter size for the conv layers')
    parser.add_argument('--dont_save',              action='store_true',                        help='don\'t save training model during testing')
    parser.add_argument('--devices',                default='0',                type=str,       help='list of gpu device(s)')
        
    args=parser.parse_args()

#    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

    # Seed random number
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    
    dataset         = args.dataset
    batch_size      = args.batch_size
    architecture    = args.architecture
    learning_rate   = args.learning_rate
    pretrained_ann  = args.pretrained_ann
    epochs          = args.epochs
    lr_reduce       = args.lr_reduce
    optimizer       = args.optimizer
    dropout         = args.dropout
    kernel_size     = args.kernel_size

    values = args.lr_interval.split()
    lr_interval = []
    for value in values:
        lr_interval.append(int(float(value)*args.epochs))
    
    
   
        
    # Training settings
    if torch.cuda.is_available() and args.gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    # Loading Dataset
    if dataset == 'CIFAR100':
        normalize   = transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))
        labels      = 100 
    elif dataset == 'CIFAR10':
        normalize   = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        labels      = 10
    elif dataset == 'MNIST':
        labels = 10
    elif dataset == 'IMAGENET':
        normalize   = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        labels = 1000

    
    if dataset == 'CIFAR10' or dataset == 'CIFAR100':
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
        ])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])
    
    if dataset == 'CIFAR100':
        train_dataset   = datasets.CIFAR100(root='./cifar_data', train=True, download=True,transform =transform_train)
        test_dataset    = datasets.CIFAR100(root='./cifar_data', train=False, download=True, transform=transform_test)
    
    elif dataset == 'CIFAR10': 
        train_dataset   = datasets.CIFAR10(root='./cifar_data', train=True, download=True,transform =transform_train)
        test_dataset    = datasets.CIFAR10(root='./cifar_data', train=False, download=True, transform=transform_test)
    
    elif dataset == 'MNIST':
        train_dataset   = datasets.MNIST(root='./mnist/', train=True, download=True, transform=transforms.ToTensor()
            )
        test_dataset    = datasets.MNIST(root='./mnist/', train=False, download=True, transform=transforms.ToTensor())
    elif dataset == 'IMAGENET':
        traindir    = os.path.join('/local/a/imagenet/imagenet2012/', 'train')
        valdir      = os.path.join('/local/a/imagenet/imagenet2012/', 'val')
        train_dataset    = datasets.ImageFolder(
                            traindir,
                            transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize,
                            ]))
        test_dataset     = datasets.ImageFolder(
                            valdir,
                            transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                normalize,
                            ]))

    
    train_loader    = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    test_loader     = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    if architecture[0:3].lower() == 'vgg':
        model = VGG(vgg_name=architecture, labels=labels, dataset=dataset, kernel_size=kernel_size, dropout=dropout)
        model1=VGG_bn_bias(vgg_name=architecture, labels=labels, dataset=dataset, kernel_size=kernel_size, dropout=dropout)
    elif architecture[0:3].lower() == 'res':
        if architecture.lower() == 'resnet12':
            model = ResNet12(labels=labels, dropout=dropout)
        elif architecture.lower() == 'resnet20':
            model = ResNet20(labels=labels, dropout=dropout)
        elif architecture.lower() == 'resnet34':
            model = ResNet34(labels=labels, dropout=dropout) 
    #f.write('\n{}'.format(model))
    
    #CIFAR100 sometimes has problem to start training
    #One solution is to train for CIFAR10 with same architecture
    #Load the CIFAR10 trained model except the final layer weights
    model = nn.DataParallel(model)
    model1=nn.DataParallel(model1)
    pretrained_ann  = './trained_models/'+'ann_vgg16_imagenet_bs128_acc70.86.pth'
#    pretrained_ann  ='/home/nano01/a/chowdh23/vgg9_snn_surrgrad_backprop/CHECKPOINTS/'+'ann_vgg16_cifar10_bs64_sgd_drop.2_lr.01_.5red_bn_acc94.64.pth'
    
    if pretrained_ann:
        state=torch.load(pretrained_ann, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(state['state_dict'], strict=False)
        print('\n Missing keys : {}, Unexpected Keys: {}'.format(missing_keys, unexpected_keys))
        
    
    #print('\n {}'.format(model)) 
    co=0
    
    for l in range(len(model.module.features)):
            if isinstance(model.module.features[l], nn.Conv2d):
                # here beta and gamma are used oppositely, ie beta in actual formula is gamma here, vice-versa
                conv=model.module.features[l]
                bn=model.module.features[l+1]
                w = conv.weight
                mean = bn.running_mean
                var_sqrt = torch.sqrt(bn.running_var + bn.eps)
                beta = bn.weight
                gamma = bn.bias
                if conv.bias is not None:
                    b = conv.bias
                else:
                    b = mean.new_zeros(mean.shape)
                w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
                b = (b - mean)/var_sqrt * beta + gamma
#                fused_conv = nn.Conv2d(conv.in_channels,
#                         conv.out_channels,
#                         conv.kernel_size,
#                         conv.stride,
#                         conv.padding,
#                         bias=True)
                model1.module.features[co].weight = nn.Parameter(w)
                model1.module.features[co].bias = nn.Parameter(b)
                co+=3
                #print(b.max())
    for l in range(len(model.module.classifier)):
            if isinstance(model.module.classifier[l], nn.Linear):
                model1.module.classifier[l].weight = nn.Parameter(model.module.classifier[l].weight)
                
                
    state = {
#                    'accuracy'      : max_accuracy,
#                    'epoch'         : epoch,
                    'state_dict'    : model1.state_dict(),
#                    'optimizer'     : optimizer.state_dict()
            }
    
#    filename = '/home/nano01/a/chowdh23/vgg9_snn_surrgrad_backprop/CHECKPOINTS/'+'ann_vgg16_cifar10_bn_fused'+'.pth'
    
    filename ='./trained_models/'+'ann_vgg16_imagenet_bn_fused'+'.pth'
    torch.save(state,filename)
    
    
    
#    print('\n {}'.format(model1))
##    f=len(model1.features)
##    for l in range(f):
##            if isinstance(model1.features[l], nn.BatchNorm2d):
##                del model1.features[l]
                
    
                



