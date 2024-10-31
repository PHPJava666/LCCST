
import torch.nn as nn
import torch.nn.functional as F
import torch

def conv3d_output_size(input_size, kernel_size, stride, padding):
    return (input_size - kernel_size + 2 * padding) // stride + 1


class Discriminator(nn.Module):#从SSASNet中借鉴来的
    def __init__(self, input_channels, num_classes):
        super(Discriminator, self).__init__()

        self.conv0 = nn.Conv3d(input_channels, 64, kernel_size=4, stride=2, padding=1)
        # self.conv0 = nn.Conv3d(num_classes, 64, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(64, 64*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv3d(64*2, 64*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv3d(64*4, 64*8, kernel_size=4, stride=2, padding=1)
        self.avgpool = nn.AvgPool3d((7, 7, 5))
        self.classifier = nn.Linear(64*8, 2)

        # self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=False)
        self.dropout = nn.Dropout3d(0.5)
        self.Softmax = nn.Softmax()

    def forward(self, map, image):
        batch_size = map.shape[0]#input_map = torch.randn(4, 1, 112, 112, 80)所以batch_sizes是4
        map_feature = self.conv0(map)#input_image = torch.randn(4, 1, 112, 112, 80)这里的输入map应该是
        #map的输入尺寸应该是
        image_feature = self.conv1(image)
        x = torch.add(map_feature, image_feature)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.leaky_relu(x)

        x = self.avgpool(x)

        x = x.view(batch_size, -1)
        x = self.classifier(x)
        x = x.reshape((batch_size, 2))
        
        return x


        # self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=3, stride=2, padding=1)
        # self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)

        # # Calculate the size of the input to the first fully connected layer
        # self._initialize_fc_input_size(num_classes)

        # self.fc1 = nn.Linear(128 * (num_classes // 4) * (num_classes // 4) * (num_classes // 4), 1024)
        # self.fc2 = nn.Linear(1024, 1)
        # self.relu = nn.ReLU(inplace=True)
        # self.sigmoid = nn.Sigmoid()

    # def _initialize_fc_input_size(self, input_size):
    #     size = conv3d_output_size(input_size, kernel_size=3, stride=2, padding=1)
    #     size = conv3d_output_size(size, kernel_size=3, stride=2, padding=1)
    #     self.fc_input_size = 128 * size * size * size

    # def forward(self, x):
    #     x = self.relu(self.conv1(x))
    #     x = self.relu(self.conv2(x))
    #     x = x.view(x.size(0), -1)
    #     x = self.relu(self.fc1(x))
    #     x = self.sigmoid(self.fc2(x))
    #     return x


# 测试功能的主函数
if __name__ == '__main__':
    # compute FLOPS & PARAMETERS
    from thop import profile
    from thop import clever_format

    # Initialize the model
    model = Discriminator(input_channels=1, num_classes=2)

    # Create dummy inputs with matching shapes
    input_map = torch.randn(4, 1, 112, 112, 80)
    input_image = torch.randn(4, 1, 112, 112, 80)

    #
    outputs = model(input_map, input_image)#这里就需要用到forword里面的参数了
    print(outputs)
    print(outputs.shape)#[4,2]

    # Compute FLOPS and parameters
    flops, params = profile(model, inputs=(input_map,input_image))
    macs, params = clever_format([flops, params], "%.3f")
    print(macs, params)

    # import ipdb; ipdb.set_trace()