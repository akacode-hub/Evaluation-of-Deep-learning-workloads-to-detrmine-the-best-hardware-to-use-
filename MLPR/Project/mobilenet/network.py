from torch import nn
import torch 

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

class ConvReLUBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        padding = (kernel_size - 1) // 2
        super(ConvReLUBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

class Model(nn.Module):
    def __init__(self, num_layers_lst, num_channels_lst, num_classes):
        super(Model, self).__init__()

        self.gating_layers = []
        self.num_classes = num_classes
        self.num_channels_lst = num_channels_lst
        self.fc = nn.Linear(self.num_channels_lst[-1]*4, self.num_classes)

        for i in range(7):
            self.gating_layers.append(ConvBNReLU(num_channels_lst[i], num_channels_lst[i+1], kernel_size=3, stride=2))

            for j in range(num_layers_lst[i]):
                self.gating_layers.append(ConvBNReLU(num_channels_lst[i+1], num_channels_lst[i+1], kernel_size=3, stride=1))

        self.gate_network = nn.Sequential(*self.gating_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.gate_network(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)    

        return x

class Model1(nn.Module):
    def __init__(self, num_layers_lst, num_channels_lst, num_layers, num_classes):
        super(Model1, self).__init__()

        self.gating_layers = []
        self.num_classes = num_classes
        self.num_channels_lst = num_channels_lst
        self.num_layers = num_layers
        self.fc1 = nn.Linear(self.num_channels_lst[self.num_layers]*7*7, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

        for i in range(self.num_layers):
            self.gating_layers.append(ConvBNReLU(num_channels_lst[i], num_channels_lst[i+1], kernel_size=3, stride=2))

            for j in range(num_layers_lst[i]):
                self.gating_layers.append(ConvBNReLU(num_channels_lst[i+1], num_channels_lst[i+1], kernel_size=3, stride=1))

        self.gate_network = nn.Sequential(*self.gating_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.gate_network(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)    
        x = self.fc2(x)
        x = self.fc3(x)

        return x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()

        self.gating_layers = []
        self.gating_layers.append(ConvReLUBN(3, 32, kernel_size=3, stride=1))
        self.gating_layers.append(ConvReLUBN(32, 32, kernel_size=3, stride=2))
        self.gating_layers.append(nn.Dropout(0.3))

        self.gating_layers.append(ConvReLUBN(32, 64, kernel_size=3, stride=1))
        self.gating_layers.append(ConvReLUBN(64, 64, kernel_size=3, stride=2))
        self.gating_layers.append(nn.Dropout(0.3))

        self.gating_layers.append(ConvReLUBN(64, 128, kernel_size=3, stride=1))
        self.gating_layers.append(ConvReLUBN(128, 128, kernel_size=3, stride=2))
        self.gating_layers.append(nn.Dropout(0.5))

        self.gating_layers.append(Flatten())

        self.gating_layers.append(nn.Linear(14*14*128, 512))
        self.gating_layers.append(nn.ReLU6(inplace=True))
        self.gating_layers.append(nn.BatchNorm1d(512))
        self.gating_layers.append(nn.Dropout(0.5))

        self.gating_layers.append(nn.Linear(512, 128))
        self.gating_layers.append(nn.ReLU6(inplace=True))
        self.gating_layers.append(nn.Dropout(0.25))

        self.gating_layers.append(nn.Linear(128, 10))

        self.gate_network = nn.Sequential(*self.gating_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.gate_network(x)

        return x

if __name__ == "__main__":

    num_layers_lst = [1, 1, 1, 1, 1, 1, 1]
    num_channels_lst = [3, 16, 32, 64, 128, 256]
    num_classes = 10
    num_layers = 4
    # network = Model1(num_layers_lst, num_channels_lst, num_layers, num_classes)
    network = Model2()
    from torchsummary import summary
    summary(network.cuda(), (3, 112, 112))
