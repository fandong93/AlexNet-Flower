
import json
import torch
import torchvision.transforms as transforms
from model import AlexNet
import matplotlib.pyplot as plt
from PIL import Image


def main():
    # 数据增广方法
    dataset_transforms = transforms.Compose([transforms.Resize([224, 224]),  # 随机裁剪至 224 x 224
                                             transforms.ToTensor(),  # 转换至 Tensor
                                             transforms.Normalize(0.5, 0.5)])

    img = Image.open("./flower_photos/roses/0pred.jpeg")
    # 显示图片
    plt.figure("Image")
    plt.imshow(img)
    plt.axis('off')  # 关掉坐标轴为 off
    plt.title("Rose")
    plt.show()

    img = dataset_transforms(img)
    img = torch.reshape(img, (1, 3, 224, 224))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AlexNet().to(device)
    model.load_state_dict(torch.load('./model/AlexNet-Flower.pth'))  # 加载模型
    model.eval()  # 把模型转为 valid 模式

    with open('class_indices.json', 'r') as json_file:
        classes = json.load(json_file)
        class_list = list(classes.values())

    # 预测
    output = model(img.to(device))
    predict = output.argmax(dim=1)
    pred_class = class_list[predict.item()]
    print("预测类别：", pred_class)
    print("roses" == pred_class)


if __name__ == '__main__':
    main()
