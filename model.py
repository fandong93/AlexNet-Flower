
import torch.nn as nn


class AlexNet(nn.Module):
    # 五层卷积，三层全连接（输入图片大小是 C x H x W  ---> 3 * 64 * 64）
    def __init__(self, num_classes=5, init_weights=False):
        super(AlexNet, self).__init__()
        # 五个卷积层
        self.con1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),      # input[3, 224, 224]  output[96, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                      # output[96, 27, 27]
        )
        self.con2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),     # output[256, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                      # output[256, 13, 13]
        )
        self.con3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),    # output[384, 13, 13]
            nn.ReLU(inplace=True),
        )
        self.con4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),    # output[384, 13, 13]
            nn.ReLU(inplace=True),
        )
        self.con5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),    # output[256, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                      # output[256, 6, 6]
        )

        self.features = nn.Sequential(
            self.con1,
            self.con2,
            self.con3,
            self.con4,
            self.con5,
        )

        # 三层全连接
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, input):
        featurea = self.features(input)
        featurea = featurea.view(featurea.shape[0], -1)
        outputs = self.classifier(featurea)
        return outputs

    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                # 使用正态分布对输入张量进行赋值
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Linear):
                # 使 layer.weight 值服从正态分布 N(mean, std)，默认值为 0，1。通常设置较小的值。
                nn.init.normal_(layer.weight, 0, 0.01)
                # 使 layer.bias 值为常数 val
                nn.init.constant_(layer.bias, 0)


# # 测试
# model = AlexNet(init_weights=True)
# img =torch.rand([64, 3, 224, 224])
# outputs = model.forward(img)
# print(outputs.shape)
