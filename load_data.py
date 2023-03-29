
import os
import json
import torch.utils.data as data
from torchvision import transforms, datasets
from split_data import Split


class Load:
    def loader(self, train_size, valid_size, nw):
        print('Using {} dataloader workers every process'.format(nw))

        split = Split()
        split.splitdata("./flower_photos", "./datasets")

        data_transform = {
            "train": transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
            "valid": transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

        # os.getcwd() 方法用于返回当前工作目录
        # os.path.abspath() 方法用于返回绝对路径
        data_root = os.path.abspath(os.path.join(os.getcwd()))  # ../.. 表示上一级的上一级
        image_path = os.path.join(data_root, "datasets")
        assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

        train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"), transform=data_transform["train"])
        train_loader = data.DataLoader(train_dataset, batch_size=train_size, shuffle=True, num_workers=nw)
        train_num = len(train_dataset)

        flower_list = train_dataset.class_to_idx    # 按顺序为这些类别定义索引为 0，1 ...
        # print(flower_list)  # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
        cla_dict = dict((val, key) for key, val in flower_list.items())
        # json.dumps 用于将 python 对象编码成 json 字符串
        json_str = json.dumps(cla_dict, indent=4)   # indent 的值代表缩进空格数量
        with open('class_indices.json', 'w') as json_file:
            json_file.write(json_str)

        valid_dataset = datasets.ImageFolder(root=os.path.join(image_path, "valid"), transform=data_transform["valid"])
        valid_loader = data.DataLoader(valid_dataset, batch_size=valid_size, shuffle=False, num_workers=nw)
        valid_num = len(valid_dataset)

        print("using {} images for training, {} images for validation.".format(train_num, valid_num))
        return train_loader, valid_loader


# # 测试
# load = Load()
# load.loader(64, 100, 2)
