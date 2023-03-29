
import os
import torch
import time
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model import AlexNet
import train_module
import valid_module
from load_data import Load


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    load = Load()
    train_loader, test_loader = load.loader(64, 100, 4)

    model = AlexNet(num_classes=5, init_weights=True).to(device)

    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epochs = 50
    best_acc = 0.0
    max_loss = 0.0
    min_loss = 1.0
    max_acc = 0.0
    min_acc = 1.0

    save_path = './model/AlexNet-Flower.pth'
    if not os.path.exists("./model"):
        os.mkdir("./model")

    Loss = []
    Accuracy = []

    train = train_module.Train()
    valid = valid_module.Valid()

    print("start_time", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    for epoch in range(epochs + 1):
        loss, train_acc = train.train_method(model, device, train_loader, loss_function, optimizer, epoch)

        if loss > max_loss:
            max_loss = loss
        if loss < min_loss:
            min_loss = loss

        if train_acc > max_acc:
            max_acc = train_acc
        if train_acc < min_acc:
            min_acc = train_acc

        test_acc = valid.valid_method(model, device, test_loader, epoch)
        Loss.append(loss)
        Accuracy.append(train_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)  # pytorch 中的 state_dict 是一个简单的 python 的字典对象，将每一层与它的对应参数建立映射关系。

    print("end_time: ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    print('Finished Training')
    print('best_acc: ', best_acc)

    plt.subplot(2, 1, 1)
    plt.plot(Loss)
    plt.title('Loss')
    x_ticks = torch.arange(0, epochs + 1, 10)
    plt.xticks(x_ticks)
    y_ticks = torch.arange(min_loss, max_loss, 0.3)
    plt.yticks(y_ticks)

    plt.subplot(2, 1, 2)
    plt.plot(Accuracy)
    plt.title('Accuracy')
    x_ticks = torch.arange(0, epochs + 1, 10)
    plt.xticks(x_ticks)
    y_ticks = torch.arange(min_acc, max_acc, 0.1)
    plt.yticks(y_ticks)

    plt.subplots_adjust(hspace=0.3)  # 调整子图间距

    if not os.path.exists("./img"):
        os.mkdir("./img")

    plt.savefig('./img/AlexNet-Flower.jpg')
    plt.show()


if __name__ == '__main__':
    main()
