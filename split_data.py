
import os
from shutil import copy
import random


class Split:
    def mkfile(self, file):
        if not os.path.exists(file):
            os.makedirs(file)

    def splitdata(self, file_path, datasets_path):
        # 获取 flower_photos 文件夹下除 .txt 文件以外所有文件夹名（即 5 种花的类名）
        flower_class = [cla for cla in os.listdir(file_path) if ".txt" not in cla]  # not in：右侧的内容中是否不包含左侧的内容

        # 创建 训练集 train 文件夹，并由5种类名在其目录下创建 5 个子目录
        train_path = datasets_path + '/train/'
        self.mkfile(train_path)
        for cla in flower_class:
            self.mkfile(train_path + cla)

        # 创建 验证集 test 文件夹，并由 5 种类名在其目录下创建 5 个子目录
        valid_path = datasets_path + '/valid/'
        self.mkfile(valid_path)
        for cla in flower_class:
            self.mkfile(valid_path + cla)

        # 划分比例，训练集 : 验证集 = 9 : 1
        split_rate = 0.1
        # 遍历 5 种花的全部图像并按比例分成训练集和验证集
        for cla in flower_class:
            cla_path = file_path + '/' + cla + '/'  # 某一类别花的子目录
            images = os.listdir(cla_path)  # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
            num = len(images)
            # sample() 函数返回一个从序列即列表、元组、字符串或集合中选择的特定长度的项目列表
            eval_index = random.sample(images, k=int(num * split_rate))  # 从 images 列表中随机抽取 k 个图像
            for index, image in enumerate(images):
                # eval_index 中保存验证集 test 的图像名称
                if image in eval_index:
                    image_path = cla_path + image
                    new_path = valid_path + cla
                    copy(image_path, new_path)  # 将选中的图像复制到新路径
                # 其余的图像保存在训练集 train 中
                else:
                    image_path = cla_path + image
                    new_path = train_path + cla
                    copy(image_path, new_path)
                # \r 表示将光标的位置回退到本行的开头位置
                print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")  # processing bar
            print()
        print("processing done!")


# # 测试
# split = Split()
# split.splitdata("./flower_photos", "./datasets")
