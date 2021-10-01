import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import math

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, dropout):
        #print('In __init__ BasicBlock')
        #super(BasicBlock, self).__init__()
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            )
        self.identity = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.identity = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        #print('In forward BasicBlock')
        out = self.residual(x) + self.identity(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        #super(Bottleneck, self).__init__()
        #print('In __init__ Bottleneck')
        super().__init__()
        self.delay_path = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
            nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.expansion*planes),
            )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        #print('In forward Bottleneck')
        out = self.delay_path(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
        
    def __init__(self, block, num_blocks, labels=10, dropout=0.2):
        #print('In __init__ ResNet')
        super(ResNet, self).__init__()

        self.in_planes      = 64
        self.dropout        = dropout
        self.pre_process    = nn.Sequential(
                                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),
                                nn.Dropout(self.dropout),
                                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),
                                nn.Dropout(self.dropout),
                                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),
                                nn.AvgPool2d(2)
                                )
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, dropout=self.dropout)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, dropout=self.dropout)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, dropout=self.dropout)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, dropout=self.dropout)
        #self.avgpool = nn.AvgPool2d(2)
        self.classifier     = nn.Sequential(
                                nn.Linear(512*2*2, labels, bias=False)
                                )
        self._initialize_weights2()

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
        
    def _make_layer(self, block, planes, num_blocks, stride, dropout):
        #print('\n In _make_layer')
        if num_blocks==0:
            return nn.Sequential()
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dropout))
            self.in_planes = planes * block.expansion
        #pdb.set_trace()
        return nn.Sequential(*layers)

    def forward(self, x):
        #print('In forward ResNet')
        
        out = self.pre_process(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = self.avgpool(out)
        out = out.view(x.size(0), -1)
        out = self.classifier(out)
        return out


#def ResNet8(labels=10):
#    return ResNet(BasicBlock, [1,1,0,0], labels)

def ResNet12(labels=10, dropout=0.2):
    return ResNet(block=BasicBlock, num_blocks=[1,1,1,1], labels=labels, dropout=dropout)

def ResNet20(labels=10, dropout=0.2):
    return ResNet(block=BasicBlock, num_blocks=[2,2,2,2], labels=labels, dropout=dropout)

def ResNet34(labels=10, dropout=0.2):
    return ResNet(block=BasicBlock, num_blocks=[3,4,5,3], labels=labels, dropout=dropout)

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    print('In test()')
    net = ResNet34()
    print('Calling y=net() from test()')
    y = net(torch.randn(1,3,32,32))
    print(y.size())

if __name__ == '__main__':
    test()