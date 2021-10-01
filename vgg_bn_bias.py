import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import math


cfg = {
    'VGG4' : [64, 'A', 128, 'A'],
    'VGG5' : [64, 'A', 128, 128, 'A'],
    'VGG9':  [64, 'A', 128, 256, 'A', 256, 512, 'A', 512, 'A', 512],
    'VGG11': [64, 'A', 128, 256, 'A', 512, 512, 'A', 512, 'A', 512, 512],
    'VGG13': [64, 64, 'A', 128, 128, 'A', 256, 256, 'A', 512, 512, 512, 'A', 512],
    'VGG16': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 'A', 512, 512, 512, 'A', 512, 512, 512],
    'VGG19': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 256, 'A', 512, 512, 512, 512, 'A', 512, 512, 512, 512]
}


class VGG_bn_bias(nn.Module):
    def __init__(self, vgg_name='VGG16', labels=10, dataset = 'CIFAR10', kernel_size=3, dropout=0.2):
        super(VGG_bn_bias, self).__init__()
        
        self.dataset        = dataset
        self.kernel_size    = kernel_size
        self.dropout        = dropout
        self.features       = self._make_layers(cfg[vgg_name])

        if vgg_name == 'VGG5' and dataset!= 'MNIST':
            self.classifier = nn.Sequential(
                            nn.Linear(512*4*4, 4096, bias=False),
                            nn.ReLU(inplace=True),
                            nn.Dropout(self.dropout),
                            nn.Linear(4096, 4096, bias=False),
                            nn.ReLU(inplace=True),
                            nn.Dropout(self.dropout),
                            nn.Linear(4096, labels, bias=False)
                            )
        elif vgg_name == 'VGG4' and dataset== 'MNIST':
            self.classifier = nn.Sequential(
                            nn.Linear(128*7*7, 1024, bias=False),
                            nn.ReLU(inplace=True),
                            nn.Dropout(self.dropout),
                            #nn.Linear(4096, 4096, bias=False),
                            #nn.ReLU(inplace=True),
                            #nn.Dropout(0.5),
                            nn.Linear(1024, labels, bias=False)
                            )
        elif vgg_name!='VGG5' and dataset!='MNIST':
            self.classifier = nn.Sequential(
                            nn.Linear(512*2*2, 4096, bias=False),
                            nn.ReLU(inplace=True),
                            nn.Dropout(self.dropout),
                            nn.Linear(4096, 4096, bias=False),
                            nn.ReLU(inplace=True),
                            nn.Dropout(self.dropout),
                            nn.Linear(4096, labels, bias=False)
                            )
        elif vgg_name == 'VGG5' and dataset == 'MNIST':
            self.classifier = nn.Sequential(
                            nn.Linear(128*7*7, 4096, bias=False),
                            nn.ReLU(inplace=True),
                            nn.Dropout(self.dropout),
                            nn.Linear(4096, 4096, bias=False),
                            nn.ReLU(inplace=True),
                            nn.Dropout(self.dropout),
                            nn.Linear(4096, labels, bias=False)
                            )
        elif vgg_name!='VGG5' and dataset =='MNIST':
            self.classifier = nn.Sequential(
                            nn.Linear(512*1*1, 4096, bias=False),
                            nn.ReLU(inplace=True),
                            nn.Dropout(self.dropout),
                            nn.Linear(4096, 4096, bias=False),
                            nn.ReLU(inplace=True),
                            nn.Dropout(self.dropout),
                            nn.Linear(4096, labels, bias=False)
                            )
        
        self._initialize_weights2()
        

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _initialize_weights2(self):
        for m in self.modules():
            
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def _make_layers(self, cfg):
        layers = []

        if self.dataset == 'MNIST':
            in_channels = 1
        else:
            in_channels = 3
        
        for x in cfg:
            stride = 1
            
            if x == 'A':
                layers.pop()
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2, stride=stride, bias=True),
                           #nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)
                           ]
                layers += [nn.Dropout(self.dropout)]           
                in_channels = x

        
        return nn.Sequential(*layers)

def test():
    for a in cfg.keys():
        if a=='VGG5' or a=='VGG4':
            continue
        net = VGG_bn_bias(a)
        x = torch.randn(2,3,32,32)
        y = net(x)
        print(y.size())
    # For VGG5 change the linear layer in self. classifier from '512*2*2' to '512*4*4'    
    # net = VGG('VGG5')
    # x = torch.randn(2,3,32,32)
    # y = net(x)
    # print(y.size())
if __name__ == '__main__':
    test()
