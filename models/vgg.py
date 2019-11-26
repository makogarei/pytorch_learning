import torch
import torch.nn.functional as F
import numpy as np
from glob import glob

class VGG16(torch.nn.Module):
    def __init__(self,num_classes,channel):
        super(VGG16, self).__init__()
        self.num_classes = num_classes
        self.channel = channel

        #論文でいうとDの実装に当たる
        #3channel_224_224
        self.conv1_1 = torch.nn.Conv2d(self.channel, 64, kernel_size=3, padding=1, stride=1)
        self.conv1_2 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.conv2_1 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        self.conv2_2 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)
        self.conv3_1 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1)
        self.conv3_2 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
        self.conv3_3 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
        self.conv4_1 = torch.nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1)
        self.conv4_2 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1)
        self.conv4_3 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1)
        self.conv5_1 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1)
        self.conv5_2 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1)
        self.conv5_3 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1)
        
        #25088=512*7*7
        #本来なら25088だがCIFAR10は32*32しかないので、512に設定
        #Channel*7Height*7Width
        #https://qiita.com/mathlive/items/d9f31f8538e20a102e14
        #http://www.sanko-shoko.net/note.php?id=pyk1

        # self.fc1 = torch.nn.Linear(25088, 4096)
        self.fc1 = torch.nn.Linear(512, 4096)
        self.fc2 = torch.nn.Linear(4096, 4096)
        self.fc_out = torch.nn.Linear(4096, self.num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.max_pool2d(x, 2, stride=2, padding=0)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.max_pool2d(x, 2, stride=2, padding=0)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = F.max_pool2d(x, 2, stride=2, padding=0)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = F.max_pool2d(x, 2, stride=2, padding=0)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = F.max_pool2d(x, 2, stride=2, padding=0)
    
        #全結合層に入る前に.viewで1次元に落とす
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.nn.Dropout()(x)
        x = F.relu(self.fc2(x))
        x = torch.nn.Dropout()(x)
        x = self.fc_out(x)
        x = F.softmax(x, dim=1)
        
        return x
