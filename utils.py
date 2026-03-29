# 导入 PyTorch 相关模块
import torch
from torch.utils.data import DataLoader, Dataset  # 数据加载器和数据集类
from torch.utils.data.sampler import Sampler  # 采样器类
import numpy as np  # 数值计算库
import scipy as sp  # 科学计算库
import scipy.stats  # 科学计算中的统计模块
import random  # Python 内置的随机数模块
import scipy.io as sio  # 用于读取和写入 MATLAB 文件
from sklearn import preprocessing  # 用于数据预处理的工具
import matplotlib.pyplot as plt  # 用于绘图
from torchvision import transforms  # 用于图像预处理的工具
import torch.nn as nn
import torch.nn.functional as F


# 设置默认的计算设备（优先使用 GPU，否则使用 CPU）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义一个函数，用于设置随机种子，确保实验的可重复性
def same_seeds(seed):
    torch.manual_seed(seed)  # 设置 PyTorch 的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 设置 CUDA 的随机种子
        torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 的随机种子（多 GPU 环境）
    np.random.seed(seed)  # 设置 NumPy 的随机种子
    random.seed(seed)  # 设置 Python 内置随机模块的种子
    torch.backends.cudnn.benchmark = False  # 禁用 cuDNN 的基准模式
    torch.backends.cudnn.deterministic = True  # 设置 cuDNN 为确定性模式

# 定义一个函数，用于计算均值和置信区间
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)  # 将输入数据转换为 NumPy 数组
    n = len(a)  # 获取数据的长度（样本数量）
    m, se = np.mean(a), scipy.stats.sem(a)  # 计算均值和标准误差
    h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)  # 计算置信区间的半宽
    return m, h  # 返回均值和置信区间半宽

# 从 operator 模块导入 truediv 函数，用于安全地进行除法运算
from operator import truediv

# 定义一个函数，用于计算混淆矩阵的平均准确率（AA）和每类的准确率
def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]  # 获取混淆矩阵的维度（类别数量）
    list_diag = np.diag(confusion_matrix)  # 提取混淆矩阵的对角线元素（每类的正确预测数量）
    list_raw_sum = np.sum(confusion_matrix, axis=1)  # 计算每行的和（每类的总预测数量）
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))  # 计算每类的准确率，避免除零错误
    average_acc = np.mean(each_acc)  # 计算平均准确率（AA）
    return each_acc, average_acc  # 返回每类的准确率和平均准确率


import torch.utils.data as data  # 导入 PyTorch 的数据工具模块

# 定义一个自定义的数据集类 matcifar，继承自 PyTorch 的 Dataset 类
class matcifar(data.Dataset):
    """
    `CIFAR10 <<url id="cv5b24jacc49nht5259g" type="url" status="parsed" title="CIFAR-10 and CIFAR-100 datasets" wc="9273">https://www.cs.toronto.edu/~kriz/cifar.html</url> >`_ Dataset.
    Args:
        imdb (dict): 数据集的字典形式，包含数据和标签。
        train (bool, optional): 如果为 True，则创建训练集；否则创建测试集。
        d (int, optional): 数据维度参数，用于调整数据的形状。
        medicinal (int, optional): 用于选择数据组织方式的标志。
    """

    # 初始化函数，设置数据集的基本参数
    def __init__(self, imdb, train, d, medicinal):
        self.train = train  # 标记当前是训练集还是测试集
        self.imdb = imdb  # 数据集的字典形式，包含数据和标签
        self.d = d  # 数据维度参数
        self.x1 = np.argwhere(self.imdb['set'] == 1)  # 找到训练集的索引
        self.x2 = np.argwhere(self.imdb['set'] == 3)  # 找到测试集的索引
        self.x1 = self.x1.flatten()  # 将训练集索引展平为一维数组
        self.x2 = self.x2.flatten()  # 将测试集索引展平为一维数组

        # 根据 medicinal 参数选择数据组织方式
        if medicinal == 1:
            # 如果 medicinal 为 1，使用第一种数据组织方式
            self.train_data = self.imdb['data'][self.x1, :, :, :]  # 提取训练数据
            self.train_labels = self.imdb['Labels'][self.x1]  # 提取训练标签
            self.test_data = self.imdb['data'][self.x2, :, :, :]  # 提取测试数据
            self.test_labels = self.imdb['Labels'][self.x2]  # 提取测试标签
        else:
            # 如果 medicinal 不为 1，使用第二种数据组织方式
            self.train_data = self.imdb['data'][:, :, :, self.x1]  # 提取训练数据
            self.train_labels = self.imdb['Labels'][self.x1]  # 提取训练标签
            self.test_data = self.imdb['data'][:, :, :, self.x2]  # 提取测试数据
            self.test_labels = self.imdb['Labels'][self.x2]  # 提取测试标签

            # 根据 d 参数调整数据的形状
            if self.d == 3:
                self.train_data = self.train_data.transpose((3, 2, 0, 1))  # 调整训练数据的维度顺序
                self.test_data = self.test_data.transpose((3, 2, 0, 1))  # 调整测试数据的维度顺序
            else:
                self.train_data = self.train_data.transpose((3, 2, 0, 1))  # 调整训练数据的维度顺序
                self.test_data = self.test_data.transpose((3, 2, 0, 1))  # 调整测试数据的维度顺序

    # 定义获取数据项的函数
    def __getitem__(self, index):
        """
        Args:
            index (int): 索引
        Returns:
            tuple: (image, target) 其中 target 是目标类别的索引。
        """
        # 根据是否为训练集，选择对应的数据和标签
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # 返回图像和目标标签
        return img, target

    # 定义获取数据集长度的函数
    def __len__(self):
        # 返回数据集的长度，根据是训练集还是测试集
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

# 定义一个函数，用于检查数据集中每个类别的样本数量是否满足最小要求
def sanity_check(all_set):
    nclass = 0  # 初始化类别计数器
    nsamples = 0  # 初始化样本计数器
    all_good = {}  # 初始化一个字典，用于存储满足条件的类别和样本

    # 遍历数据集中的每个类别
    for class_ in all_set:
        if len(all_set[class_]) >= 200:  # 检查当前类别的样本数量是否大于等于 200
            all_good[class_] = all_set[class_][:200]  # 选择前 200 个样本
            nclass += 1  # 类别计数器加 1
            nsamples += len(all_good[class_])  # 累加样本数量

    # 打印满足条件的类别和样本数量
    print('the number of class:', nclass)
    print('the number of sample:', nsamples)
    return all_good  # 返回满足条件的类别和样本

# 定义一个函数，用于对数据进行翻转扩展
def flip(data):
    y_4 = np.zeros_like(data)  # 创建一个与输入数据形状相同的零矩阵
    y_1 = y_4  # 复制零矩阵
    y_2 = y_4  # 复制零矩阵
    first = np.concatenate((y_1, y_2, y_1), axis=1)  # 水平拼接三个零矩阵
    second = np.concatenate((y_4, data, y_4), axis=1)  # 水平拼接零矩阵、原始数据和零矩阵
    third = first  # 复制第一行的零矩阵
    Data = np.concatenate((first, second, third), axis=0)  # 垂直拼接三行矩阵
    return Data  # 返回扩展后的数据

# 定义一个函数，用于加载和预处理光谱图像数据
def load_data(image_file, label_file):
    image_data = sio.loadmat(image_file)  # 使用 scipy.io.loadmat 加载图像文件
    label_data = sio.loadmat(label_file)  # 使用 scipy.io.loadmat 加载标签文件

    # 提取文件名作为键名
    data_key = image_file.split('/')[-1].split('.')[0]  # 提取图像文件名（去除路径和扩展名）
    label_key = label_file.split('/')[-1].split('.')[0]  # 提取标签文件名（去除路径和扩展名）

    # 根据键名从字典中提取数据
    data_all = image_data[data_key]  # 提取图像数据（形状为 [行, 列, 波段]）
    GroundTruth = label_data[label_key]  # 提取标签数据（形状为 [行, 列]）

    # 获取数据的维度信息
    [nRow, nColumn, nBand] = data_all.shape  # 提取图像的行数、列数和波段数
    print(data_key, nRow, nColumn, nBand)  # 打印数据的基本信息

    # 将三维数据展平为二维矩阵（形状为 [像素数, 波段数]）
    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))

    # 对数据进行标准化处理（减去均值，除以标准差）
    data_scaler = preprocessing.scale(data)

    # 将标准化后的数据重新调整为三维形状
    Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1], data_all.shape[2])

    # 返回预处理后的图像数据和标签数据
    return Data_Band_Scaler, GroundTruth  # 返回标准化后的图像数据和标签数据

# 定义一个函数，用于对数据进行随机裁剪和缩放
def Crop_and_resize(data):
    da = transforms.RandomResizedCrop(9, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333))  # 创建随机裁剪和缩放的变换
    data = data.transpose(2, 0, 1)  # 调整数据的维度顺序，以适配 PyTorch 的输入格式 (C, H, W)
    x = da(torch.from_numpy(data))  # 应用随机裁剪和缩放变换
    x = x.numpy()  # 将数据从 Tensor 转换为 NumPy 数组
    x = x.transpose(1, 2, 0)  # 将维度顺序调整回原始格式 (H, W, C)
    return x  # 返回处理后的数据

# 定义一个函数，用于对数据进行随机翻转增强
def flip_augmentation(data):  # 输入数据的形状为 (H, W, C)
    horizontal = np.random.random() > 0.5  # 随机决定是否进行水平翻转
    vertical = np.random.random() > 0.5  # 随机决定是否进行垂直翻转
    if horizontal:  # 如果水平翻转标志为 True
        data = np.fliplr(data)  # 对数据进行水平翻转
    if vertical:  # 如果垂直翻转标志为 True
        data = np.flipud(data)  # 对数据进行垂直翻转
    return data  # 返回增强后的数据

# 定义一个函数，用于对数据进行辐射噪声增强
def radiation_noise(data):
    argument = (np.random.randint(1, 10), np.random.randint(1, 10))  # 随机生成裁剪尺寸
    da_1 = transforms.CenterCrop(argument)  # 创建中心裁剪的变换
    da_2 = transforms.Resize((9, 9))  # 创建缩放变换，将裁剪后的数据缩放到固定大小 (9, 9)
    data = data.transpose(2, 0, 1)  # 调整数据的维度顺序，以适配 PyTorch 的输入格式 (C, H, W)
    x = da_1(torch.from_numpy(data))  # 应用中心裁剪变换
    x = da_2(x)  # 应用缩放变换
    x = x.numpy()  # 将数据从 Tensor 转换为 NumPy 数组
    x = x.transpose(1, 2, 0)  # 将维度顺序调整回原始格式 (H, W, C)
    return x  # 返回处理后的数据

# 定义一个类，用于创建少样本学习任务
class Task(object):
    def __init__(self, data, num_classes, shot_num, query_num):
        self.data = data  # 输入的数据集，通常是一个字典，键为类别，值为样本列表
        self.num_classes = num_classes  # 任务中的类别数量
        self.support_num = shot_num  # 每个类别的支持集样本数量
        self.query_num = query_num  # 每个类别的查询集样本数量

        # 获取所有类别的列表，并随机选择 num_classes 个类别
        class_folders = sorted(list(data))  # 获取所有类别的列表
        class_list = random.sample(class_folders, self.num_classes)  # 随机选择 num_classes 个类别

        # 为每个类别分配一个标签
        labels = np.array(range(len(class_list)))  # 创建一个从 0 到 num_classes-1 的标签数组
        labels = dict(zip(class_list, labels))  # 将类别名称映射到标签

        # 初始化样本字典
        samples = dict()

        # 初始化支持集和查询集的数据和标签列表
        self.support_datas = []  # 支持集数据
        self.query_datas = []  # 查询集数据
        self.support_labels = []  # 支持集标签
        self.query_labels = []  # 查询集标签
        self.support_real_labels = []
        self.query_real_labels = []

        # 遍历每个类别，随机选择支持集和查询集样本
        for c in class_list:
            temp = self.data[c]  # 获取当前类别的所有样本
            samples[c] = random.sample(temp, len(temp))  # 随机打乱样本顺序
            random.shuffle(samples[c])  # 再次打乱样本顺序（确保随机性）

            # 添加支持集样本和标签
            self.support_datas += samples[c][:shot_num]  # 选择前 shot_num 个样本作为支持集
            self.support_labels += [labels[c] for i in range(shot_num)]  # 添加对应的标签

            # 添加查询集样本和标签
            self.query_datas += samples[c][shot_num:shot_num + query_num]  # 选择接下来的 query_num 个样本作为查询集
            self.query_labels += [labels[c] for i in range(query_num)]  # 添加对应的标签

            # 分别存储支持集和查询集的样本真实标签（类别名称形式）。
            self.support_real_labels += [c for i in range(shot_num)]
            self.query_real_labels += [c for i in range(query_num)]

# 定义一个抽象的少样本数据集类
class FewShotDataset(Dataset):
    def __init__(self, task, split='train'):
        self.task = task  # 任务对象，包含支持集和查询集的数据和标签
        self.split = split  # 数据集类型（训练集或测试集）
        # 根据 split 参数选择支持集或查询集的数据和标签
        self.image_datas = self.task.support_datas if self.split == 'train' else self.task.query_datas
        self.labels = self.task.support_labels if self.split == 'train' else self.task.query_labels

    def __len__(self):
        return len(self.image_datas)  # 返回数据集的长度（样本数量）

    def __getitem__(self, idx):
        # 抽象方法，子类需要实现具体的 __getitem__ 方法
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")

# 定义一个具体的少样本数据集类，继承自 FewShotDataset
class HBKC_dataset(FewShotDataset):
    def __init__(self, *args, **kwargs):
        super(HBKC_dataset, self).__init__(*args, **kwargs)  # 调用父类的构造函数

    def __getitem__(self, idx):
        image = self.image_datas[idx]  # 获取指定索引的图像数据
        label = self.labels[idx]  # 获取对应的标签
        return image, label  # 返回图像和标签

# 定义一个类平衡采样器
class ClassBalancedSampler(Sampler):
    '''
    Samples 'num_inst' examples each from 'num_cl' pool of examples of size 'num_per_class'
    '''
    def __init__(self, num_per_class, num_cl, num_inst, shuffle=True):
        self.num_per_class = num_per_class  # 每个类别的样本数量
        self.num_cl = num_cl  # 类别数量
        self.num_inst = num_inst  # 每个类别的实例数量
        self.shuffle = shuffle  # 是否随机打乱样本顺序

    def __iter__(self):
        # 根据是否打乱样本顺序，生成采样索引
        if self.shuffle:
            # 随机打乱样本顺序
            batch = [[i + j * self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        else:
            # 按顺序生成样本索引
            batch = [[i + j * self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        # 将嵌套列表展平为一维列表
        batch = [item for sublist in batch for item in sublist]

        # 如果需要打乱顺序，则随机打乱
        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)  # 返回迭代器

    def __len__(self):
        return 1  # 返回采样器的长度（固定为 1）

# 定义一个函数，用于获取少样本数据集的数据加载器
def get_HBKC_data_loader(task, num_per_class=1, split='train', shuffle=False):
    dataset = HBKC_dataset(task, split=split)  # 创建数据集对象

    # 根据 split 参数选择支持集或查询集的采样器
    if split == 'train':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.support_num, shuffle=shuffle)  # 支持集
    else:
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.query_num, shuffle=shuffle)  # 查询集

    # 创建数据加载器
    loader = DataLoader(dataset, batch_size=19 * task.num_classes, sampler=sampler)
    # loader = DataLoader(dataset, batch_size=108, sampler=sampler)
    return loader  # 返回数据加载器

# 定义一个函数，用于绘制分类图
def classification_map(map, groundTruth, dpi, savePath):
    fig = plt.figure(frameon=False)  # 创建无边框的图形
    # 设置图形大小
    fig.set_size_inches(groundTruth.shape[1] * 2.0 / dpi, groundTruth.shape[0] * 2.0 / dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])  # 创建一个覆盖整个图形的坐标轴
    ax.set_axis_off()  # 关闭坐标轴
    ax.xaxis.set_visible(False)  # 隐藏 x 轴
    ax.yaxis.set_visible(False)  # 隐藏 y 轴
    fig.add_axes(ax)  # 将坐标轴添加到图形中

    ax.imshow(map)  # 显示分类图
    fig.savefig(savePath, dpi=dpi)  # 保存图形到指定路径
    return 0  # 返回 0，表示成功

# 定义欧几里得距离度量函数
def euclidean_metric(a, b):
    """
    计算两个特征之间的欧几里得距离。
    :param a: 第一个特征
    :param b: 第二个特征
    :return: 欧几里得距离
    """
    n = a.shape[0]  #获取第一个特征 a 的样本数量。
    m = b.shape[0]  #获取第一个特征 b 的样本数量。
    a = a.unsqueeze(1).expand(n, m, -1)  # 扩展维度  在 a 的第1维（索引为0）上添加一个维度，并扩展到 (n, m, -1) 的形状。-1 表示保持原来的特征维度不变。
    b = b.unsqueeze(0).expand(n, m, -1)  # 扩展维度  在 b 的第0维（索引为0）上添加一个维度，并扩展到 (n, m, -1) 的形状。
    logits = -((a - b) ** 2).sum(dim=2)  # 计算 a 和 b 之间的欧几里得距离。首先计算 a 和 b 的差的平方，然后在第2维（特征维度）上求和，最后取负值。
    return logits

# def adjust_feature_dimensions(text_features, target_batch_size):
#     """
#     调整文本特征的维度以匹配目标批量大小。
#
#     参数:
#     text_features -- 输入的文本特征张量，形状为 [batch_size, 1, feature_dim]
#     target_batch_size -- 目标批量大小
#
#     返回:
#     调整后的文本特征张量，形状为 [target_batch_size, 1, feature_dim]
#     """
#     text_features = text_features.permute(1, 0, 2)
#     # 获取原始批量大小和特征维度
#     original_batch_size, _, feature_dim = text_features.size()
#
#     # 计算需要重复的次数
#     repeat_times = target_batch_size // original_batch_size + 1
#
#     # 重复文本特征以匹配目标批量大小
#     # 使用 repeat 方法在第0维（批量维度）上重复特征
#     adjusted_features = text_features.repeat(repeat_times, 1, 1)
#
#     # 如果重复后批量大小超过目标批量大小，则裁剪多余的部分
#     if adjusted_features.size(0) > target_batch_size:
#         adjusted_features = adjusted_features[:target_batch_size]
#
#     # adjusted_features = adjusted_features.permute(1, 0, 2)
#
#     return adjusted_features
#
# class ImageTextContrastiveLoss(nn.Module):
#     def __init__(self, temperature=0.07):
#         super(ImageTextContrastiveLoss, self).__init__()
#         self.temperature = temperature
#         # 使用全连接层调整文本特征的维度
#         # self.conv = nn.Linear(128, 100)
#
#     def forward(self, image_features, text_features):
#         # text_features = self.conv(text_features)
#         # 计算图像特征和文本特征之间的相似度
#         image_features = image_features.unsqueeze(1)
#         text_features = text_features.unsqueeze(0)
#         text_features = adjust_feature_dimensions(text_features, 128)
#         # print(image_features.shape)
#         # print(text_features.shape)
#         similarity = F.cosine_similarity(image_features, text_features, dim=2)
#         # print(similarity.shape)
#         # 应用温度缩放
#         similarity = similarity / self.temperature
#
#         # 创建一个掩码，将相同样本的相似度设置为负无穷
#         batch_size = image_features.shape[0]
#         mask = torch.eye(batch_size, dtype=torch.bool).to(device)
#
#         # 计算损失
#         losses = -torch.log(similarity.diag() / similarity.sum(dim=1))
#         return losses.mean()
#

class ImageTextContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cuda', temperature=0.1):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))  # 超参数 温度
        self.register_buffer("negatives_mask",
                             (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())

    def forward(self, emb_i, emb_j):

        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = euclidean_metric(representations, representations)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)

        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)

        return loss
