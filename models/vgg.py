import torch.nn as nn

class My_VGG(nn.Module):
    def __init__(self,num_of_classes):
        super(My_VGG,self).__init__()
        self.features = nn.Sequential(
            #input (1,28,28)
            nn.Conv2d(1,64,kernel_size=3),
            #(64,26,26)
            nn.Conv2d(64,128,kernel_size=3),
            #(128,24,24)
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),
            #(128,12,12)

            nn.Conv2d(128,256,kernel_size=3),
            #(256,10,10)
            nn.Conv2d(256,512,kernel_size=3),
            #(512,8,8)
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2),
            #(512,4,4)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512*4*4,1024),
            nn.Dropout(),
            nn.Linear(1024,512),
            nn.Dropout(),
            nn.Linear(512,num_of_classes),
        )

    def forward(self,x):
        in_size = x.size(0)
        x = self.features(x)
        x = x.view(in_size,-1)
        x = self.classifier(x)
        return x