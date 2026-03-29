# 导入 PyTorch 相关模块
import torch
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 神经网络功能模块
from torch.autograd import Variable  # 自动求导模块
from torch.optim.lr_scheduler import StepLR  # 学习率调度器
from torch.utils.data import DataLoader, Dataset  # 数据加载和数据集模块
from torch.utils.data.sampler import Sampler  # 数据采样器模块

# 导入其他常用库
import numpy as np  # 数值计算库
import os  # 操作系统接口模块
import math  # 数学函数库
import argparse  # 命令行参数解析模块
import scipy as sp  # 科学计算库
import scipy.stats  # 统计模块
import pickle  # 数据序列化模块
import random  # 随机数生成模块
import scipy.io as sio  # 用于读取 MATLAB 文件的模块
from sklearn.decomposition import PCA  # 主成分分析模块
from sklearn import metrics  # 评估指标模块
import matplotlib.pyplot as plt  # 数据可视化库
from scipy.io import loadmat  # 用于加载 MATLAB 文件的模块
from sklearn import preprocessing  # 数据预处理模块
from sklearn.neighbors import KNeighborsClassifier  # K近邻分类器
from matplotlib import pyplot  # 数据可视化库
from matplotlib.colors import ListedColormap  # 自定义颜色映射模块
import time  # 时间模块
import utils  # 自定义工具模块（假设 utils 是一个自定义模块）
import sys  # 系统相关功能模块
from torchvision import transforms  # 图像预处理模块
from vit import Transformer  # 自定义 Transformer 模块（假设 vit 是一个自定义模块）
import vit


# 创建 ArgumentParser 对象，用于定义脚本的描述信息
parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")

# 添加命令行参数，定义每个参数的名称、类型、默认值和帮助信息
# 特征维度，默认值为160
parser.add_argument("-f", "--feature_dim", type=int, default=160, help="Dimension of the feature space")
# 源域输入维度，默认值为128
parser.add_argument("-c", "--src_input_dim", type=int, default=128, help="Dimension of the input data in the source domain")
# 目标域输入维度，默认值为200
parser.add_argument("-d", "--tar_input_dim", type=int, default=103, help="Dimension of the input data in the target domain")
# N 维度，默认值为100
parser.add_argument("-n", "--n_dim", type=int, default=100, help="Dimension of the N space")
# 类别数量，默认值为16
parser.add_argument("-w", "--class_num", type=int, default=9, help="Number of classes")
# 每个类别的支持样本数量，默认值为1
parser.add_argument("-s", "--shot_num_per_class", type=int, default=1, help="Number of support samples per class")
# 每个类别的查询样本数量，默认值为19
parser.add_argument("-b", "--query_num_per_class", type=int, default=19, help="Number of query samples per class")
# 训练轮数，默认值为20000
parser.add_argument("-e", "--episode", type=int, default=20000, help="Number of training episodes")
# 测试轮数，默认值为600
parser.add_argument("-t", "--test_episode", type=int, default=600, help="Number of testing episodes")
# 学习率，默认值为0.001
parser.add_argument("-l", "--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
# 使用的 GPU 编号，默认值为4
parser.add_argument("-g", "--gpu", type=int, default=4, help="GPU device number to use")
# 隐藏单元数量，默认值为10
parser.add_argument("-u", "--hidden_unit", type=int, default=10, help="Number of hidden units in the network")

# 目标域相关参数
# 目标域类别数量，默认值为16
parser.add_argument("-m", "--test_class_num", type=int, default=9, help="Number of classes in the target domain")
# 目标域每个类别的标记样本数量，默认值为5
parser.add_argument("-z", "--test_lsample_num_per_class", type=int, default=5, help="Number of labeled samples per class in the target domain")
# Transformer 层的数量，默认值为2
parser.add_argument("-a", "--trans_layer", type=int, default=2, help="Number of Transformer layers")

# 解析命令行参数，将所有参数存储在 args 对象中
args = parser.parse_args()

# 检查是否有可用的 GPU，并设置设备为 "cuda:0"（第一个 GPU）或 "cpu"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 将解析后的参数赋值给对应的变量，以便在后续代码中使用
# 特征维度
FEATURE_DIM = args.feature_dim
# 源域输入维度
SRC_INPUT_DIMENSION = args.src_input_dim
# 目标域输入维度
TAR_INPUT_DIMENSION = args.tar_input_dim
# N 维度
N_DIMENSION = args.n_dim
# 类别数量
CLASS_NUM = args.class_num
# 每个类别的支持样本数量
SHOT_NUM_PER_CLASS = args.shot_num_per_class
# 每个类别的查询样本数量
QUERY_NUM_PER_CLASS = args.query_num_per_class
# 训练轮数
EPISODE = args.episode
# 测试轮数
TEST_EPISODE = args.test_episode
# 学习率
LEARNING_RATE = args.learning_rate
# 使用的 GPU 编号
GPU = args.gpu
# 隐藏单元数量
HIDDEN_UNIT = args.hidden_unit

# 目标域相关参数
# 目标域类别数量
TEST_CLASS_NUM = args.test_class_num
# 目标域每个类别的标记样本数量
TEST_LSAMPLE_NUM_PER_CLASS = args.test_lsample_num_per_class


# 定义一个函数来设置随机种子，确保实验的可重复性
def seed_torch(seed):
    random.seed(seed)  # 设置 Python 的随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置环境变量，确保 Python 的 hash 随机性
    np.random.seed(seed)  # 设置 NumPy 的随机种子
    torch.manual_seed(seed)  # 设置 PyTorch 的随机种子（CPU）
    torch.cuda.manual_seed(seed)  # 设置 PyTorch 的随机种子（单 GPU）
    torch.cuda.manual_seed_all(seed)  # 设置 PyTorch 的随机种子（多 GPU）
    torch.backends.cudnn.benchmark = False  # 禁用 cuDNN 的基准测试模式
    torch.backends.cudnn.deterministic = True  # 设置 cuDNN 的确定性模式

# 调用函数，设置随机种子为 0
seed_torch(0)

# get src/tar class number -> label semantic vector
# labels_sou = ["water", "bare soil school", "bare soil park", "bare soil farmland", "natural plants", "weeds in farmland", "forest", "grass", "rice field grown", "rice field first stage", "row crops", "plastic house", "manmade non dark", "manmade dark", "manmade blue", "manmade red", "manmade grass", "asphalt","paved ground"]

# IP
# labels_tar = ["Alfalfa", "Corn notill", "Corn mintill", "Corn", "Grass pasture", "Grass trees", "Grass pasture mowed", "Hay windrowed", "Oats", "Soybean notill", "Soybean mintill", "Soybean clean", "Wheat", "Woods", "Buildings Grass Trees Drives", "Stone Steel Towers"]

# salinas
# labels_tar = ["Brocoli green weeds 1", "Brocoli green weeds 2", "Fallow", "Fallow rough plow", "Fallow smooth", "Stubble", "Celery", "Grapes untrained", "Soil vinyard develop", "Corn senesced green weeds","Lettuce romaine 4wk", "Lettuce romaine 5wk", "Lettuce romaine 6wk", "Lettuce romaine 7wk" , "Vinyard untrained", "Vinyard vertical trellis"]

#UP
# labels_tar = ["Asphalt", "Meadows", "Gravel", "Trees", "Sheets", "Bare soil", "Bitumen", "Bricks", "Shadow"]

# get src/tar class number -> label semantic vector
# labels_sou_all = ["water, bare soil school, bare soil park, bare soil farmland, natural plants, weeds in farmland, forest, grass, rice field grown, rice field first stage, row crops, plastic house, manmade non dark, manmade dark, manmade blue, manmade red, manmade grass, asphalt,paved ground"]
# labels_sou_all = ["water"]
# IP
# labels_tar_all = ["Alfalfa, Corn notill, Corn mintill, Corn, Grass pasture, Grass trees, Grass pasture mowed, Hay windrowed, Oats, Soybean notill, Soybean mintill, Soybean clean, Wheat, Woods, Buildings Grass Trees Drives, Stone Steel Towers"]
# labels_tar_all = ["Alfalfa"]
# salinas
# labels_tar_all = ["Brocoli green weeds 1, Brocoli green weeds 2, Fallow, Fallow rough plow, Fallow smooth, Stubble, Celery, Grapes untrained, Soil vinyard develop, Corn senesced green weeds,Lettuce romaine 4wk, Lettuce romaine 5wk, Lettuce romaine 6wk, Lettuce romaine 7wk , Vinyard untrained, Vinyard vertical trellis"]

#UP
# labels_tar_all = ["Asphalt, Meadows, Gravel, Trees, Sheets, Bare soil, Bitumen, Bricks, Shadow"]
# from transformers import BertModel, BertTokenizer
# model = BertModel.from_pretrained('/data/yuchao/data/yuchao/FFSL/pretrain-model/bert-base-uncased')
# model.eval()
# tokenizer = BertTokenizer.from_pretrained('/data/yuchao/data/yuchao/FFSL/pretrain-model/bert-base-uncased')
# labels_sou = [
#     "water: Large areas of water bodies, such as lakes, rivers, or oceans.",
#     "bare soil school: Bare soil areas found within school premises, potentially from construction or activity fields.",
#     "bare soil park: Bare soil spots in parks, commonly seen in walkways or recreational zones.",
#     "bare soil farmland: Bare soil sections in agricultural fields, likely due to plowing or fallow land.",
#     "natural plants: Vegetation that grows naturally, such as forests, grasslands, or wild flora.",
#     "weeds in farmland: Weeds present in crop fields, often competing with cultivated plants for resources.",
#     "forest: Dense woodland areas, comprising a variety of trees and understory vegetation.",
#     "grass: Grassy areas, which could be natural grasslands or artificial turf.",
#     "rice field grown: Rice fields in the growth stage, where the rice plants have developed to a certain height.",
#     "rice field first stage: The initial stage of rice fields, where rice plants are just beginning to grow.",
#     "row crops: Crops planted in rows, such as corn, soybeans, or other cultivated plants.",
#     "plastic house: Greenhouses or hoop houses covered with plastic sheeting for temperature control.",
#     "manmade non dark: Man-made structures that are not dark in color, like white or light-colored buildings.",
#     "manmade dark: Man-made structures that are dark in color, such as black or dark grey buildings.",
#     "manmade blue: Man-made structures painted in blue, potentially indicating specific usage or ownership.",
#     "manmade red: Man-made structures painted in red, which might signify particular functions or affiliations.",
#     "manmade grass: Artificial turf, often used for sports fields or decorative purposes.",
#     "asphalt: Paved surfaces made of asphalt, commonly found in roads or parking lots.",
#     "paved ground: Ground covered with paving materials, such as concrete or brick for sidewalks."
# ]
# #UP
# labels_tar = [
#     "Asphalt: A black cementitious material used for paving roads, parking lots, and airport runways.",
#     "Meadows: Open areas covered with grass and often wildflowers, typically found in rural or natural settings.",
#     "Gravel: A surface covered with small stones or pebbles, commonly used for driveways and walkways.",
#     "Trees: Areas dominated by large plants with woody stems and leaves, forming a forest or wooded area.",
#     "Sheets: Flat, thin layers, possibly referring to metal sheets, plastic sheets, or other similar materials.",
#     "Bare soil: Soil that is exposed without vegetation cover, often seen in construction sites or eroded areas.",
#     "Bitumen: A sticky, black, and highly viscous liquid or semi-solid form of petroleum used for paving and roofing.",
#     "Bricks: Solid blocks of clay or concrete, commonly used as building materials for walls and structures.",
#     "Shadow: Dark areas produced by the obstruction of light, often seen under objects or structures."
# ]
#
# labels_sou_all = [
#     "water: Large areas of water bodies, such as lakes, rivers, or oceans,\
#     bare soil school: Bare soil areas found within school premises, potentially from construction or activity fields,\
#     bare soil park: Bare soil spots in parks, commonly seen in walkways or recreational zones,\
#     bare soil farmland: Bare soil sections in agricultural fields, likely due to plowing or fallow land,\
#     natural plants: Vegetation that grows naturally, such as forests, grasslands, or wild flora,\
#     weeds in farmland: Weeds present in crop fields, often competing with cultivated plants for resources,\
#     forest: Dense woodland areas, comprising a variety of trees and understory vegetation,\
#     grass: Grassy areas, which could be natural grasslands or artificial turf,\
#     rice field grown: Rice fields in the growth stage, where the rice plants have developed to a certain height,\
#     rice field first stage: The initial stage of rice fields, where rice plants are just beginning to grow,\
#     row crops: Crops planted in rows, such as corn, soybeans, or other cultivated plants,\
#     plastic house: Greenhouses or hoop houses covered with plastic sheeting for temperature control,\
#     manmade non dark: Man-made structures that are not dark in color, like white or light-colored buildings,\
#     manmade dark: Man-made structures that are dark in color, such as black or dark grey buildings,\
#     manmade blue: Man-made structures painted in blue, potentially indicating specific usage or ownership,\
#     manmade red: Man-made structures painted in red, which might signify particular functions or affiliations,\
#     manmade grass: Artificial turf, often used for sports fields or decorative purposes,\
#     asphalt: Paved surfaces made of asphalt, commonly found in roads or parking lots,\
#     paved ground: Ground covered with paving materials, such as concrete or brick for sidewalks."
# ]
# #UP
# labels_tar_all = [
#     "Asphalt: A black cementitious material used for paving roads, parking lots, and airport runways,\
#     Meadows: Open areas covered with grass and often wildflowers, typically found in rural or natural settings,\
#     Gravel: A surface covered with small stones or pebbles, commonly used for driveways and walkways,\
#     Trees: Areas dominated by large plants with woody stems and leaves, forming a forest or wooded area,\
#     Sheets: Flat, thin layers, possibly referring to metal sheets, plastic sheets, or other similar materials,\
#     Bare soil: Soil that is exposed without vegetation cover, often seen in construction sites or eroded areas,\
#     Bitumen: A sticky, black, and highly viscous liquid or semi-solid form of petroleum used for paving and roofing,\
#     Bricks: Solid blocks of clay or concrete, commonly used as building materials for walls and structures,\
#     Shadow: Dark areas produced by the obstruction of light, often seen under objects or structures."
# ]

labels_sou = [
    "water: Large areas of water bodies, such as lakes, rivers, or oceans.",
    "bare soil school: Bare soil areas found within school premises, potentially from construction or activity fields.",
    "bare soil park: Exposed ground sections in parks, commonly seen in walkways or recreational zones.",
    "bare soil farmland: Uncovered earth patches in agricultural fields, likely due to plowing or fallow land.",
    "natural plants: Vegetation that grows naturally, such as forests, grasslands, or wild flora.",
    "weeds in farmland: Weeds present in crop fields, often competing with cultivated plants for resources.",
    "forest: Dense woodland areas, comprising a variety of trees and understory vegetation.",
    "grass: Grassy areas, which could be natural grasslands or artificial turf.",
    "rice field grown: Rice fields in the growth stage, where the rice plants have developed to a certain height.",
    "rice field first stage: The initial stage of rice fields, where rice plants are just beginning to grow.",
    "row crops: Crops planted in rows, such as corn, soybeans, or other cultivated plants.",
    "plastic house: Greenhouses or hoop houses covered with plastic sheeting for temperature control.",
    "manmade non dark: Artificial constructions with a light hue, like white or light-colored buildings.",
    "manmade dark: Man-made structures that are dark in color, such as black or dark grey buildings.",
    "manmade blue: Constructed edifices coated in azure, potentially indicating specific usage or ownership.",
    "manmade red: Synthetic constructions coated with red paint, which might signify particular functions or affiliations.",
    "manmade grass: Human-made turf, often used for sports fields or decorative purposes.",
    "asphalt: Paved surfaces made of asphalt, commonly found in roads or parking lots.",
    "paved ground: Ground covered with paving materials, such as concrete or brick for sidewalks."
]
#UP
labels_tar = [
    "Asphalt: A black cementitious material used for paving roads, parking lots, and airport runways.",
    "Meadows: Open areas covered with grass and often wildflowers, typically found in rural or natural settings.",
    "Gravel: A surface covered with small stones or pebbles, commonly used for driveways and walkways.",
    "Trees: Areas dominated by large plants with woody stems and leaves, forming a forest or wooded area.",
    "Sheets: Flat, thin layers, possibly referring to metal sheets, plastic sheets, or other similar materials.",
    "Bare soil: Soil that is exposed without vegetation cover, often seen in construction sites or eroded areas.",
    "Bitumen: A sticky, black, and highly viscous liquid or semi-solid form of petroleum used for paving and roofing.",
    "Bricks: Solid blocks of clay or concrete, usually used as building materials for walls and structures.",
    "Shadow: Dark areas produced by the obstruction of light, often seen under objects or structures."
]

labels_sou_all = [
    "water: Large areas of water bodies, such as lakes, rivers, or oceans,\
    bare soil school: Bare soil areas found within school premises, potentially from construction or activity fields,\
    bare soil park: Exposed ground sections in parks, commonly seen in walkways or recreational zones,\
    bare soil farmland: Uncovered earth patches in agricultural fields, likely due to plowing or fallow land,\
    natural plants: Vegetation that grows naturally, such as forests, grasslands, or wild flora,\
    weeds in farmland: Weeds present in crop fields, often competing with cultivated plants for resources,\
    forest: Dense woodland areas, comprising a variety of trees and understory vegetation,\
    grass: Grassy areas, which could be natural grasslands or artificial turf,\
    rice field grown: Rice fields in the growth stage, where the rice plants have developed to a certain height,\
    rice field first stage: The initial stage of rice fields, where rice plants are just beginning to grow,\
    row crops: Crops planted in rows, such as corn, soybeans, or other cultivated plants,\
    plastic house: Greenhouses or hoop houses covered with plastic sheeting for temperature control,\
    manmade non dark: Artificial constructions with a light hue, like white or light-colored buildings,\
    manmade dark: Man-made structures that are dark in color, such as black or dark grey buildings,\
    manmade blue: Constructed edifices coated in azure, potentially indicating specific usage or ownership,\
    manmade red: Synthetic constructions coated with red paint, which might signify particular functions or affiliations,\
    manmade grass: Human-made turf, often used for sports fields or decorative purposes,\
    asphalt: Paved surfaces made of asphalt, commonly found in roads or parking lots,\
    paved ground: Ground covered with paving materials, such as concrete or brick for sidewalks."
]
#UP
labels_tar_all = [
    "Asphalt: A black cementitious material used for paving roads, parking lots, and airport runways,\
    Meadows: Open areas covered with grass and often wildflowers, typically found in rural or natural settings,\
    Gravel: A surface covered with small stones or pebbles, commonly used for driveways and walkways,\
    Trees: Areas dominated by large plants with woody stems and leaves, forming a forest or wooded area,\
    Sheets: Flat, thin layers, possibly referring to metal sheets, plastic sheets, or other similar materials,\
    Bare soil: Soil that is exposed without vegetation cover, often seen in construction sites or eroded areas,\
    Bitumen: A sticky, black, and highly viscous liquid or semi-solid form of petroleum used for paving and roofing,\
    Bricks: Solid blocks of clay or concrete, usually used as building materials for walls and structures,\
    Shadow: Dark areas produced by the obstruction of light, often seen under objects or structures."
]

from transformers import RobertaModel, RobertaTokenizer
model = RobertaModel.from_pretrained('/data/yuchao/data/yuchao/Pretrain_Model/RoBERTa')
model.eval()
tokenizer = RobertaTokenizer.from_pretrained('/data/yuchao/data/yuchao/Pretrain_Model/RoBERTa')

encoded_inputs_sou = tokenizer(labels_sou, padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    outputs_sou = model(**encoded_inputs_sou)
semantic_mapping_sou = outputs_sou.last_hidden_state[:, 0, :]  # (num_classess, 768)

encoded_inputs_tar = tokenizer(labels_tar, padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    outputs_tar = model(**encoded_inputs_tar)
semantic_mapping_tar = outputs_tar.last_hidden_state[:, 0, :]  # (num_classess, 768)

encoded_inputs_sou_all = tokenizer(labels_sou_all, padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    outputs_sou_all = model(**encoded_inputs_sou_all)
semantic_mapping_sou_all = outputs_sou_all.last_hidden_state[:, 0, :]  # (num_classess, 768)

encoded_inputs_tar_all = tokenizer(labels_tar_all, padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    outputs_tar_all = model(**encoded_inputs_tar_all)
semantic_mapping_tar_all = outputs_tar_all.last_hidden_state[:, 0, :]  # (num_classess, 768)

semantic_mapping_sou = semantic_mapping_sou.cpu().numpy()
semantic_mapping_tar = semantic_mapping_tar.cpu().numpy()
semantic_mapping_sou_all = semantic_mapping_sou_all.cpu().numpy()
semantic_mapping_tar_all = semantic_mapping_tar_all.cpu().numpy()

# print(len(semantic_mapping_sou))
# print(len(semantic_mapping_sou_all))
# print(len(semantic_mapping_tar))
# print(len(semantic_mapping_tar_all))
# 定义一个函数来初始化所需的文件夹
def _init_():
    # 如果不存在 './checkpoints' 文件夹，则创建它
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    # 如果不存在 './classificationMap' 文件夹，则创建它
    if not os.path.exists('./classificationMap'):
        os.makedirs('./classificationMap')

# 调用函数，初始化文件夹
_init_()

# 加载源域数据集
# 使用 pickle 加载源域数据集，数据集存储在指定路径的 pickle 文件中
with open(os.path.join('../datasets', '/data/yuchao/data/yuchao/Datasets/Chikusei_imdb_128.pickle'), 'rb') as handle:
    source_imdb = pickle.load(handle)  # 加载数据集
print(source_imdb.keys())  # 打印数据集的键值
print(source_imdb['Labels'])  # 打印数据集的标签

# 处理源域数据集
data_train = source_imdb['data']  # 获取训练数据，形状为 (77592, 9, 9, 128)
labels_train = source_imdb['Labels']  # 获取训练标签，长度为 77592
print(data_train.shape)  # 打印训练数据的形状
print(labels_train.shape)  # 打印训练标签的形状
keys_all_train = sorted(list(set(labels_train)))  # 获取所有类别标签并排序，例如 [0, 1, 2, ..., 18]
print(keys_all_train)  # 打印所有类别标签
label_encoder_train = {}  # 创建一个字典，用于将类别标签映射到新的索引
for i in range(len(keys_all_train)):
    label_encoder_train[keys_all_train[i]] = i  # 将每个类别标签映射到一个新的索引
print(label_encoder_train)  # 打印标签编码器

# 创建一个字典，用于存储每个类别的数据
train_set = {}
# 遍历每个样本的标签和数据
for class_, path in zip(labels_train, data_train):
    # 如果当前类别不在 train_set 中，则初始化一个空列表
    if label_encoder_train[class_] not in train_set:
        train_set[label_encoder_train[class_]] = []
    # 将当前样本的数据添加到对应类别的列表中
    train_set[label_encoder_train[class_]].append(path)
print(train_set.keys())  # 打印 train_set 的键值，即所有类别
data = train_set  # 将 train_set 赋值给 data
del train_set  # 删除 train_set 变量
del keys_all_train  # 删除 keys_all_train 变量
del label_encoder_train  # 删除 label_encoder_train 变量

# 打印源域数据集的类别数量
print("Num classes for source domain datasets: " + str(len(data)))
print(data.keys())  # 打印 data 的键值，即所有类别
# 调用 utils.sanity_check 函数，对数据进行检查，确保每个类别的样本数量大于 200
data = utils.sanity_check(data)
# 打印经过检查后，类别数量大于 200 的类别数量
print("Num classes of the number of class larger than 200: " + str(len(data)))

# 遍历每个类别的数据
for class_ in data:
    # 遍历每个样本
    for i in range(len(data[class_])):
        # 将样本的维度从 (9, 9, 128) 转置为 (128, 9, 9)
        image_transpose = np.transpose(data[class_][i], (2, 0, 1))
        # 更新 data 中的样本数据
        data[class_][i] = image_transpose

# 创建源域的 few-shot 分类数据集
metatrain_data = data
print(len(metatrain_data.keys()), metatrain_data.keys())  # 打印源域 few-shot 数据集的类别数量和类别
del data  # 删除 data 变量

# 创建源域的域适应数据集
print(source_imdb['data'].shape)  # 打印源域数据集的形状
# 将源域数据集的维度从 (77592, 9, 9, 128) 转置为 (9, 9, 128, 77592)
source_imdb['data'] = source_imdb['data'].transpose((1, 2, 3, 0))
print(source_imdb['data'].shape)  # 打印转置后的源域数据集形状
print(source_imdb['Labels'])  # 打印源域数据集的标签
# 创建源域数据集对象
source_dataset = utils.matcifar(source_imdb, train=True, d=3, medicinal=0)
# 创建源域数据加载器
source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=128, shuffle=True, num_workers=2,pin_memory=True)
# 删除不再需要的变量
del source_dataset, source_imdb


## target domain data set
# load target domain data set
test_data = '/data/yuchao/data/yuchao/Datasets/PU/paviaU.mat'  # 目标域数据集路径
test_label = '/data/yuchao/data/yuchao/Datasets/PU/paviaU_gt.mat'  # 目标域标签路径

# 使用 utils.load_data 函数加载目标域数据集和标签
Data_Band_Scaler, GroundTruth = utils.load_data(test_data, test_label)

# get train_loader and test_loader
def get_train_test_loader(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    print(Data_Band_Scaler.shape)  # 打印数据集的形状，例如 (610, 340, 103)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape  # 获取数据集的行数、列数和波段数

    '''label start'''
    num_class = int(np.max(GroundTruth))  # 获取最大类别标签
    data_band_scaler = utils.flip(Data_Band_Scaler)  # 对数据进行翻转操作
    groundtruth = utils.flip(GroundTruth)  # 对标签进行翻转操作
    del Data_Band_Scaler  # 删除原始数据集变量
    del GroundTruth  # 删除原始标签变量

    HalfWidth = 4  # 定义半宽度
    # 扩展数据集和标签的边界
    G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth]
    data = data_band_scaler[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth, :]

    [Row, Column] = np.nonzero(G)  # 获取非零元素的行和列索引
    del data_band_scaler  # 删除扩展后的数据变量
    del groundtruth  # 删除扩展后的标签变量

    nSample = np.size(Row)  # 计算非零元素的数量
    print('number of sample', nSample)  # 打印样本数量

    # Sampling samples
    train = {}  # 初始化训练集字典
    test = {}  # 初始化测试集字典
    da_train = {}  # 初始化数据增强训练集字典
    m = int(np.max(G))  # 获取最大类别标签
    nlabeled = TEST_LSAMPLE_NUM_PER_CLASS  # 获取每个类别的标记样本数量
    print('labeled number per class:', nlabeled)  # 打印每个类别的标记样本数量
    print((200 - nlabeled) / nlabeled + 1)  # 打印数据增强的倍数
    print(math.ceil((200 - nlabeled) / nlabeled) + 1)  # 打印向上取整后的数据增强倍数

    # 遍历每个类别
    for i in range(m):
        # 获取当前类别的所有索引
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if G[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)  # 随机打乱索引
        nb_val = shot_num_per_class  # 获取每个类别的支持样本数量
        train[i] = indices[:nb_val]  # 选择前 nb_val 个索引作为训练集
        da_train[i] = []  # 初始化当前类别的数据增强训练集
        # 数据增强，重复选择 nb_val 个索引多次
        for j in range(math.ceil((200 - nlabeled) / nlabeled) + 1):
            da_train[i] += indices[:nb_val]
        test[i] = indices[nb_val:]  # 剩余索引作为测试集

    # 合并所有类别的训练集、测试集和数据增强训练集索引
    train_indices = []
    test_indices = []
    da_train_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
        da_train_indices += da_train[i]
    np.random.shuffle(test_indices)  # 随机打乱测试集索引

    # 打印训练集、测试集和数据增强训练集的样本数量
    print('the number of train_indices:', len(train_indices))  # 520
    print('the number of test_indices:', len(test_indices))  # 9729
    print('the number of train_indices after data argumentation:', len(da_train_indices))  # 520
    print('labeled sample indices:', train_indices)  # 打印标记样本的索引

    # 获取训练集和测试集的样本数量
    nTrain = len(train_indices)  # 训练集样本数量
    nTest = len(test_indices)  # 测试集样本数量
    da_nTrain = len(da_train_indices)  # 数据增强后的训练集样本数量

    # 初始化一个字典，用于存储目标域数据集的相关信息
    imdb = {}
    # 初始化数据集的存储空间
    imdb['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nTrain + nTest],
                            dtype=np.float32)  # (9,9,100,n)
    imdb['Labels'] = np.zeros([nTrain + nTest], dtype=np.int64)  # 初始化标签数组
    imdb['set'] = np.zeros([nTrain + nTest], dtype=np.int64)  # 初始化数据集划分数组

    # 将训练集和测试集的索引合并
    RandPerm = train_indices + test_indices
    RandPerm = np.array(RandPerm)  # 转换为 NumPy 数组

    # 遍历所有样本，提取对应的图像块并存储到 imdb 中
    for iSample in range(nTrain + nTest):
        # 提取以当前样本为中心的图像块
        imdb['data'][:, :, :, iSample] = data[
                                         Row[RandPerm[iSample]] - HalfWidth:  Row[RandPerm[iSample]] + HalfWidth + 1,
                                         Column[RandPerm[iSample]] - HalfWidth: Column[
                                                                                    RandPerm[iSample]] + HalfWidth + 1,
                                         :]
        # 存储当前样本的标签
        imdb['Labels'][iSample] = G[Row[RandPerm[iSample]], Column[RandPerm[iSample]]].astype(np.int64)

    # 将标签调整为从 0 开始的索引
    imdb['Labels'] = imdb['Labels'] - 1  # 1-16 -> 0-15
    # 标记训练集和测试集
    imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nTest]))).astype(np.int64)
    print('Data is OK.')

    # 创建训练集和测试集的数据加载器
    train_dataset = utils.matcifar(imdb, train=True, d=3, medicinal=0)  # 创建训练集数据集对象
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=class_num * shot_num_per_class, shuffle=False,
                                               num_workers=2,pin_memory=True)  # 创建训练集数据加载器
    del train_dataset  # 删除训练集数据集对象

    test_dataset = utils.matcifar(imdb, train=False, d=3, medicinal=0)  # 创建测试集数据集对象
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2,pin_memory=True)  # 创建测试集数据加载器
    del test_dataset  # 删除测试集数据集对象
    del imdb  # 删除 imdb 字典

    # 数据增强操作，用于目标域的训练数据
    imdb_da_train = {}
    # 初始化数据增强后的数据集存储空间
    imdb_da_train['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, da_nTrain],
                                     dtype=np.float32)  # (9,9,100,n)
    imdb_da_train['Labels'] = np.zeros([da_nTrain], dtype=np.int64)  # 初始化标签数组
    imdb_da_train['set'] = np.zeros([da_nTrain], dtype=np.int64)  # 初始化数据集划分数组

    # 将数据增强后的训练集索引转换为 NumPy 数组
    da_RandPerm = np.array(da_train_indices)

    # 遍历所有数据增强后的训练样本，提取对应的图像块并存储到 imdb_da_train 中
    for iSample in range(da_nTrain):
        # 提取以当前样本为中心的图像块，并进行裁剪和调整大小操作
        imdb_da_train['data'][:, :, :, iSample] = utils.Crop_and_resize(
            data[Row[da_RandPerm[iSample]] - HalfWidth:  Row[da_RandPerm[iSample]] + HalfWidth + 1,
            Column[da_RandPerm[iSample]] - HalfWidth: Column[da_RandPerm[iSample]] + HalfWidth + 1, :])
        # 存储当前样本的标签
        imdb_da_train['Labels'][iSample] = G[Row[da_RandPerm[iSample]], Column[da_RandPerm[iSample]]].astype(np.int64)

    # 将标签调整为从 0 开始的索引
    imdb_da_train['Labels'] = imdb_da_train['Labels'] - 1  # 1-16 -> 0-15
    # 标记数据增强后的训练集
    imdb_da_train['set'] = np.ones([da_nTrain]).astype(np.int64)
    print('ok')

    # 返回训练集加载器、测试集加载器、数据增强后的训练集、原始标签、随机排列索引、行索引、列索引和训练集样本数量
    return train_loader, test_loader, imdb_da_train, G, RandPerm, Row, Column, nTrain


def get_target_dataset(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    """
    获取目标域数据集的训练集、测试集和数据增强后的训练集。
    :param Data_Band_Scaler: 目标域数据集
    :param GroundTruth: 目标域标签
    :param class_num: 类别数量
    :param shot_num_per_class: 每个类别的支持样本数量
    :return: 训练集加载器、测试集加载器、数据增强后的训练集、原始标签、随机排列索引、行索引、列索引和训练集样本数量
    """
    # 调用 get_train_test_loader 函数，获取训练集和测试集的加载器
    train_loader, test_loader, imdb_da_train, G, RandPerm, Row, Column, nTrain = get_train_test_loader(
        Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth, class_num=class_num,
        shot_num_per_class=shot_num_per_class)

    # 获取训练集的下一个批次数据
    train_datas, train_labels = train_loader.__iter__().__next__()
    print('train labels:', train_labels)
    print('size of train datas:', train_datas.shape)  # 打印训练数据的形状

    # 打印数据增强后的训练集信息
    print(imdb_da_train.keys())
    print(imdb_da_train['data'].shape)  # 打印数据增强后的训练集数据形状
    print(imdb_da_train['Labels'])  # 打印数据增强后的训练集标签

    # 删除不再需要的变量
    del Data_Band_Scaler, GroundTruth

    # 数据增强后的目标域数据
    target_da_datas = np.transpose(imdb_da_train['data'], (3, 2, 0, 1))  # 调整维度顺序
    print(target_da_datas.shape)
    target_da_labels = imdb_da_train['Labels']  # 获取数据增强后的训练集标签
    print('target data augmentation label:', target_da_labels)

    # 构建目标域的 few-shot 分类数据集
    target_da_train_set = {}
    for class_, path in zip(target_da_labels, target_da_datas):
        if class_ not in target_da_train_set:
            target_da_train_set[class_] = []
        target_da_train_set[class_].append(path)
    target_da_metatrain_data = target_da_train_set
    print(target_da_metatrain_data.keys())

    # 创建目标域的数据加载器
    target_dataset = utils.matcifar(imdb_da_train, train=True, d=3, medicinal=0)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=100, shuffle=True, num_workers=2,pin_memory=True)
    del target_dataset

    # 返回训练集加载器、测试集加载器、数据增强后的训练集、原始标签、随机排列索引、行索引、列索引和训练集样本数量
    return train_loader, test_loader, target_da_metatrain_data, target_loader, G, RandPerm, Row, Column, nTrain


# 定义一个3x3x3卷积层
def conv3x3x3(in_channel, out_channel):
    """
    定义一个3x3x3的3D卷积层，包含卷积、批量归一化。
    :param in_channel: 输入通道数
    :param out_channel: 输出通道数
    :return: 卷积层
    """
    layer = nn.Sequential(
        nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm3d(out_channel),
        # nn.ReLU(inplace=True)  # 可选的ReLU激活函数
    )
    return layer


# 定义一个映射模块
class Mapping(nn.Module):
    """
    定义一个映射模块，用于将输入特征映射到指定维度。
    :param in_dimension: 输入特征维度
    :param out_dimension: 输出特征维度
    """

    def __init__(self, in_dimension, out_dimension):
        super(Mapping, self).__init__()
        self.preconv = nn.Conv2d(in_dimension, out_dimension, kernel_size=1, stride=1, bias=False)
        self.preconv_bn = nn.BatchNorm2d(out_dimension)

    def forward(self, x):
        """
        前向传播，对输入特征进行卷积和批量归一化。
        :param x: 输入特征
        :return: 映射后的特征
        """
        x = self.preconv(x)
        x = self.preconv_bn(x)
        return x


# 定义一个映射网络
class Mapping_Network(nn.Module):
    """
    定义一个映射网络，包含两个映射模块：一个用于源域，一个用于目标域。
    """

    def __init__(self):
        super(Mapping_Network, self).__init__()
        self.target_mapping = Mapping(TAR_INPUT_DIMENSION, N_DIMENSION)  # 目标域映射
        self.source_mapping = Mapping(SRC_INPUT_DIMENSION, N_DIMENSION)  # 源域映射
        self.channel_shuffle = ChannelShuffle(10)
    def forward(self, x, domain='source'):
        """
        前向传播，根据指定的域类型选择对应的映射模块。
        :param x: 输入特征
        :param domain: 域类型，'source' 或 'target'
        :return: 映射后的特征
        """
        if domain == 'target':
            x = self.target_mapping(x)
        elif domain == 'source':
            x = self.source_mapping(x)
        x = self.channel_shuffle(x)
        return x

class ChannelShuffle(nn.Module):
    # ChannelShuffle类用于实现通道混洗操作，这是一种提高模型表达能力的技术。
    def __init__(self, group):
        super(ChannelShuffle, self).__init__()
        # group参数表示将通道分成多少组
        self.group = group

    def forward(self, x):
        # 前向传播函数，输入特征图x
        B, C, H, W = x.shape
        # 确保通道数C能被组数整除
        assert C % self.group == 0
        # 计算每组的通道数
        group_C = C // self.group
        # 将特征图x重塑为[B, group, group_C, H, W]
        x = x.view(B, self.group, group_C, H, W)
        # 交换组和通道维度，实现通道混洗
        x = x.transpose(1, 2).contiguous()
        # 将混洗后的特征图重塑回[B, -1, H, W]的形状
        x = x.view(B, C, H, W)
        return x

# 定义主网络模型
class Network(nn.Module):
    """
    定义主网络模型，包含特征提取、Transformer 编码器和分类头。
    """

    def __init__(self):
        super(Network, self).__init__()
        self.final_feat_dim = FEATURE_DIM  # 最终特征维度
        self.cls_token = nn.Parameter(torch.randn(1, 1, N_DIMENSION))  # 分类标记
        self.drop_x = nn.Dropout(0.5)  # Dropout层
        self.transformer = Transformer(N_DIMENSION, args.trans_layer, 8, 64, 1024, 0.1)  # Transformer编码器 N_DIMENSION：特征的维度。
        # args.trans_layer：Transformer 的层数 8：多头注意力机制的头数  64：每个头的维度 1024：前馈网络的隐藏维度 0.1：Dropout 比率。

    def forward(self, x):
        """
        前向传播，提取特征并通过Transformer编码器。
        :param x: 输入特征
        :return: 输出特征和注意力权重
        """
        B, C, H, W = x.shape  # 获取输入特征的形状  B 是批量大小，C 是通道数，H 和 W 是特征图的高度和宽度
        x = x.view(B, C, H * W).transpose(1, 2)  # 将输入特征展平为 (B, C, H*W)，然后调整维度顺序为 (B, H*W, C) 以便与 Transformer 的输入格式一致。
        cls_tokens = self.cls_token.repeat(x.shape[0], 1, 1)  # 将 cls_token 重复 B 次，使其形状与输入特征一致。
        x = torch.cat((x, cls_tokens), dim=1)  # 将 cls_token 拼接到输入特征的序列中，形成 (B, H*W+1, C) 的张量。
        x, att = self.transformer(x)  # 通将拼接后的特征输入到 Transformer 编码器中，获取输出特征和注意力权重。
        x = x[:, -1, :]  # 提取 Transformer 输出中对应于 cls_token 的特征，用于分类任务。
        return x, att  # 返回最终的特征和注意力权重


# 定义权重初始化函数
# 这个函数的作用是对神经网络模型的参数进行初始化。它根据模块的类型（卷积层、批量归一化层或全连接层）选择不同的初始化方法，
# 以确保模型在训练开始时具有合理的初始值。合理的初始化可以加速模型的收敛，并提高训练的稳定性。
def weights_init(m):
    """
    权重初始化函数，用于初始化模型参数。
    :param m: 模型模块
    """
    classname = m.__class__.__name__  #获取传入模块 m 的类名，用于判断模块的类型（如卷积层、批量归一化层或全连接层）。
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight, gain=1)  # Xavier均匀初始化
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)  # 正态分布初始化
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)  # Xavier正态初始化
        if m.bias is not None:
            m.bias.data = torch.ones(m.bias.data.size())


# 定义交叉熵损失函数
crossEntropy = nn.CrossEntropyLoss().cuda()
image_text_contrastive_loss = utils.ImageTextContrastiveLoss(batch_size=TEST_CLASS_NUM*19).cuda()

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


# 定义CutMix数据增强函数
def cutmix1(data, target, querys_attention, alpha=1.0):
    """
    实现CutMix数据增强，将两个图像的局部区域混合。
    :param data: 输入数据
    :param target: 目标标签
    :param querys_attention: 查询注意力权重
    :param alpha: Beta分布的参数
    :return: 混合后的数据、目标标签、混合比例
    """
    indices = torch.randperm(data.size(0))  # 随机打乱索引
    shuffled_data = data[indices]  # 打乱后的数据
    shuffled_target = target[indices]  # 打乱后的目标标签

    querys_attention = querys_attention[:, -1, 0:81]  # 获取查询注意力权重
    querys_attention = querys_attention.squeeze(dim=1).view(-1, 9, 9)  # 调整维度顺序

    lam = np.random.beta(alpha, alpha)  # 从Beta分布中采样混合比例
    lam = max(lam, 1 - lam)  # 确保混合比例在0.5到1之间

    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)  # 随机生成混合区域

    atten_part = querys_attention[:, bbx1:bbx2, bby1:bby2]  # 获取混合区域的注意力权重
    atten = torch.sum(querys_attention, axis=1)  # 计算总注意力权重
    atten = torch.sum(atten, axis=1)  # 计算总注意力权重
    atten_part = torch.sum(atten_part, axis=1)  # 计算混合区域的注意力权重
    atten_part = torch.sum(atten_part, axis=1)  # 计算混合区域的注意力权重
    lam = 1 - atten_part / atten  # 计算混合比例

    shuffled_attention = querys_attention[indices]  # 打乱后的注意力权重
    atten_part_b = shuffled_attention[:, bbx1:bbx2, bby1:bby2]  # 获取打乱后的混合区域注意力权重
    atten_b = torch.sum(shuffled_attention, axis=1)  # 计算打乱后的总注意力权重
    atten_b = torch.sum(atten_b, axis=1)  # 计算打乱后的总注意力权重
    atten_part_b = torch.sum(atten_part_b, axis=1)  # 计算打乱后的混合区域注意力权重
    atten_part_b = torch.sum(atten_part_b, axis=1)  # 计算打乱后的混合区域注意力权重
    lam_b = atten_part_b / atten_b  # 计算打乱后的混合比例

    data[:, :, bbx1:bbx2, bby1:bby2] = shuffled_data[:, :, bbx1:bbx2, bby1:bby2]  # 将混合区域替换为打乱后的数据

    # 计算输出
    target_a = target  # 原始目标标签
    target_b = shuffled_target  # 打乱后的目标标签

    return data, target_a, target_b, lam.cuda(), lam_b.cuda()

def cutmix2(data_a, target_a,querys_attention_a,data_b, target_b, querys_attention_b, alpha=1.0):
    """
    实现CutMix数据增强，将两个图像的局部区域混合。
    :param data: 输入数据
    :param target: 目标标签
    :param querys_attention: 查询注意力权重
    :param alpha: Beta分布的参数
    :return: 混合后的数据、目标标签、混合比例
    """
    indices = torch.randperm(data_b.size(0))  # 随机打乱索引
    shuffled_data_b = data_b[indices]  # 打乱后的数据
    shuffled_target_b = target_b[indices]  # 打乱后的目标标签

    querys_attention_a = querys_attention_a[:, -1, 0:81]  # 获取查询注意力权重
    querys_attention_a = querys_attention_a.squeeze(dim=1).view(-1, 9, 9)  # 调整维度顺序

    lam = np.random.beta(alpha, alpha)  # 从Beta分布中采样混合比例
    lam = max(lam, 1 - lam)  # 确保混合比例在0.5到1之间

    bbx1, bby1, bbx2, bby2 = rand_bbox(data_b.size(), lam)  # 随机生成混合区域

    atten_part_a = querys_attention_a[:, bbx1:bbx2, bby1:bby2]  # 获取混合区域的注意力权重
    atten_a = torch.sum(querys_attention_a, axis=1)  # 计算总注意力权重
    atten_a = torch.sum(atten_a, axis=1)  # 计算总注意力权重
    atten_part_a = torch.sum(atten_part_a, axis=1)  # 计算混合区域的注意力权重
    atten_part_a = torch.sum(atten_part_a, axis=1)  # 计算混合区域的注意力权重
    lam_a = 1 - atten_part_a / atten_a  # 计算混合比例

    shuffled_attention = querys_attention_b[indices]  # 打乱后的注意力权重
    atten_part_b = shuffled_attention[:, bbx1:bbx2, bby1:bby2]  # 获取打乱后的混合区域注意力权重
    atten_b = torch.sum(shuffled_attention, axis=1)  # 计算打乱后的总注意力权重
    atten_b = torch.sum(atten_b, axis=1)  # 计算打乱后的总注意力权重
    atten_part_b = torch.sum(atten_part_b, axis=1)  # 计算打乱后的混合区域注意力权重
    atten_part_b = torch.sum(atten_part_b, axis=1)  # 计算打乱后的混合区域注意力权重
    lam_b = atten_part_b / atten_b  # 计算打乱后的混合比例

    data_a[:, :, bbx1:bbx2, bby1:bby2] = shuffled_data_b[:, :, bbx1:bbx2, bby1:bby2]  # 将混合区域替换为打乱后的数据

    # 计算输出
    target_a = target_a  # 原始目标标签
    target_b = shuffled_target_b  # 打乱后的目标标签

    return data_a, target_a, target_b, lam_a.cuda(), lam_b.cuda()


# 定义一个函数，用于生成随机的边界框，用于CutMix数据增强操作
def rand_bbox(size, lam):
    W = size[2]  # 获取输入数据的宽度
    H = size[3]  # 获取输入数据的高度
    cut_rat = np.sqrt(1. - lam)  # 计算裁剪比例，基于输入的lambda值
    cut_w = int(W * cut_rat)  # 计算裁剪宽度
    cut_h = int(H * cut_rat)  # 计算裁剪高度

    # 随机选择裁剪中心点
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # 计算裁剪框的边界坐标，并确保它们不会超出图像范围
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # 返回裁剪框的四个边界坐标
    return bbx1, bby1, bbx2, bby2

# class CrossModalAttention(nn.Module):
#     def __init__(self, feat_dim=100, attn_dim=100):
#         """
#         参数说明：
#         - feat_dim: 文本和图像特征的共同输入维度 (100)
#         - attn_dim: 注意力空间的投影维度
#         """
#         super().__init__()
#         self.feat_dim = feat_dim
#         self.attn_dim = attn_dim
#         # 文本作为Query的投影层
#         self.W_q = nn.Linear(feat_dim, attn_dim).cuda()
#         # self.W_q = nn.Sequential(
#         #     nn.Linear(feat_dim, attn_dim).cuda(),
#         #     nn.ReLU()  # 增强非线性
#         # )
#         # 图像作为Key的投影层
#         self.W_k = nn.Linear(feat_dim, attn_dim).cuda()
#         # self.W_K = nn.Sequential(
#         #     nn.Linear(feat_dim, attn_dim).cuda(),
#         #     nn.ReLU()  # 增强非线性
#         # )
#         # 图像作为Value的投影层
#         self.W_v = nn.Linear(feat_dim, attn_dim).cuda()
#         # self.W_v = nn.Sequential(
#         #     nn.Linear(feat_dim, attn_dim).cuda(),
#         #     nn.ReLU()  # 增强非线性
#         # )
#
#         # 初始化参数
#         self._reset_parameters()
#
#     def _reset_parameters(self):
#         # Xavier初始化
#         nn.init.xavier_uniform_(self.W_q.weight)
#         nn.init.xavier_uniform_(self.W_k.weight)
#         nn.init.xavier_uniform_(self.W_v.weight)
#         nn.init.zeros_(self.W_q.bias)
#         nn.init.zeros_(self.W_k.bias)
#         nn.init.zeros_(self.W_v.bias)
#
#     def forward(self, t_feat, v_feat):
#         """
#         输入：
#         - t_feat: 文本特征 [batch_size, feat_dim] (304, 100)
#         - v_feat: 图像特征 [batch_size, feat_dim] (304, 100)
#
#         输出：
#         - fused_feature: 融合后的特征 [batch_size, attn_dim] (304, 256)
#         """
#         batch_size = t_feat.size(0)
#
#         # 1. 投影到公共注意力空间
#         Q = self.W_q(t_feat).cuda()  # [304, 256]
#         K = self.W_k(v_feat).cuda() # [304, 256]
#         V = self.W_v(v_feat).cuda() # [304, 256]
#
#         # 2. 计算注意力分数（单样本级别）
#         # 使用矩阵乘法实现批量计算
#         attn_scores = torch.bmm(Q.unsqueeze(1), K.unsqueeze(2))  # [304, 1, 1]
#         attn_scores = attn_scores / (100 ** 0.5)
#
#         # 3. 应用Softmax得到注意力权重
#         attn_weights = F.softmax(attn_scores, dim=-1)  # [304, 1, 1]
#
#         # 4. 加权求和（实际为缩放操作）
#         fused = attn_weights * V.unsqueeze(1)  # [304, 1, 256]
#         fused = fused.squeeze(1)  # [304, 256]
#
#         return fused

class CrossModalAttention(nn.Module):
    def __init__(self, feat_dim=100, attn_dim=100, dropout=0.1):
        super().__init__()
        self.attn_dim = attn_dim
        # 可学习投影矩阵
        self.W_q = nn.Linear(feat_dim, attn_dim).cuda()  # 文本→Query
        self.W_k = nn.Linear(feat_dim, attn_dim).cuda()  # 图像→Key
        self.W_v = nn.Linear(feat_dim, attn_dim).cuda() # 图像→Value
        self.dropout = nn.Dropout(dropout)

        # 参数初始化
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.zeros_(self.W_q.bias)
        nn.init.zeros_(self.W_k.bias)
        nn.init.zeros_(self.W_v.bias)

    def forward(self, text_feat, image_feat):
        """
        输入:
            text_feat: [batch_size, 100]
            image_feat: [batch_size, 100]
        输出:
            fused_feat: [batch_size, attn_dim]
        """
        # 1. 投影到注意力空间
        Q = self.W_q(text_feat).cuda()  # [B, attn_dim]
        K = self.W_k(image_feat).cuda()  # [B, attn_dim]
        V = self.W_v(image_feat).cuda()  # [B, attn_dim]

        # 2. 计算注意力分数
        attn_scores = torch.bmm(Q.unsqueeze(1), K.unsqueeze(2))  # [B,1,1]
        attn_scores = attn_scores / (self.attn_dim ** 0.5)

        # 3. 归一化与正则化
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)  # 防止小样本过拟合

        # 4. 加权融合
        fused = attn_weights * V.unsqueeze(1)  # [B,1,attn_dim]
        return fused.squeeze(1)  # [B, attn_dim]

# 设置运行的实验次数
nDataSet = 10
# 初始化用于存储每次实验准确率的数组
acc = np.zeros([nDataSet, 1])
# 初始化用于存储每次实验分类结果的数组
A = np.zeros([nDataSet, CLASS_NUM])
# 初始化用于存储每次实验中最佳模型的准确率
k = np.zeros([nDataSet, 1])
# 初始化用于存储所有实验中最佳模型的预测结果
best_predict_all = []
# 初始化所有实验中最佳模型的准确率
best_acc_all = 0.0
# 初始化用于存储最佳模型的相关参数
best_G, best_RandPerm, best_Row, best_Column, best_nTrain = None, None, None, None, None
# 初始化原型损失
proto_loss = torch.zeros(1)
# 定义每次实验的随机种子
# seeds = [1276, 1181, 1213, 1255, 1272, 1301, 1053, 1065, 1417, 1318]
seeds = [1276, 1213, 1255, 1272, 1301, 1053, 1065, 1126, 1365, 1382]
# seeds = [1126]
#seeds = [1330, 1220, 1336, 1337, 1224, 1236, 1226, 1235, 1233, 1229]
# 循环运行多次实验
for iDataSet in range(nDataSet):
    print('---------Start of round {} ---------'.format(iDataSet))
    # 设置当前实验的随机种子，确保实验的可重复性
    seed_torch(seeds[iDataSet])
    # 加载目标域的数据集，用于训练和测试
    train_loader, test_loader, target_da_metatrain_data, target_loader, G, RandPerm, Row, Column, nTrain = get_target_dataset(
        Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth, class_num=TEST_CLASS_NUM, shot_num_per_class=TEST_LSAMPLE_NUM_PER_CLASS)
    # 定义特征编码器网络
    feature_encoder = Network()

    semantic_feature_encoder = vit.WordEmbTransformers(feature_dim=100, dropout=0.5)
    # 定义映射网络
    mapping_network = Mapping_Network()
    # 初始化特征编码器网络的权重
    feature_encoder.apply(weights_init)
    # 初始化映射网络的权重
    mapping_network.apply(weights_init)
    semantic_feature_encoder.apply(weights_init)
    # 将特征编码器和映射网络移动到GPU上
    feature_encoder.cuda()
    mapping_network.cuda()
    semantic_feature_encoder.cuda()
    # 将特征编码器和映射网络设置为训练模式
    feature_encoder.train()
    mapping_network.train()
    semantic_feature_encoder.train()
    # 定义优化器，用于更新特征编码器和映射网络的参数
    #feature_encoder_optim = torch.optim.Adam([{'params': feature_encoder.parameters()}, {'params': mapping_network.parameters()}], lr=args.learning_rate)
    feature_encoder_optim = torch.optim.Adam([{'params': feature_encoder.parameters()}], lr=args.learning_rate)
    mapping_network_optim = torch.optim.Adam([{'params':  mapping_network.parameters()}], lr=args.learning_rate)
    semantic_feature_encoder_optim = torch.optim.Adam([{'params':  semantic_feature_encoder.parameters()}], lr=args.learning_rate)
    attn_layer = CrossModalAttention()

    # 计算参数量和FLOPS
    # from ptflops import get_model_complexity_info
    # class FullModel(nn.Module):
    #     def __init__(self, mapping, encoder):
    #         super().__init__()
    #         self.mapping = mapping
    #         self.encoder = encoder
    #
    #     def forward(self, x):
    #         x = self.mapping(x, domain='target')
    #         x, _ = self.encoder(x)
    #         return x
    #
    #
    # # 构建组合模型并计算参数量和 FLOPs（假设输入为 103x25x25）
    # model = FullModel(mapping_network, feature_encoder).cuda()
    # model.eval()
    #
    # with torch.cuda.device(0):
    #     flops, params = get_model_complexity_info(model, (103, 9, 9), as_strings=True, print_per_layer_stat=False)
    #
    # print("-------------------------------")
    # print(f"Model Params: {params}")
    # print(f"Model FLOPs: {flops}")
    # print("-------------------------------")
    # 打印开始训练的提示信息
    print("Training...")

    # 初始化上次的准确率
    last_accuracy = 0.0
    # 初始化最佳的episode
    best_episdoe = 0
    # 初始化训练损失列表
    train_loss = []
    # 初始化测试准确率列表
    test_acc = []
    # 初始化运行中的D损失、F损失、标签损失、域损失
    running_D_loss, running_F_loss = 0.0, 0.0
    running_label_loss = 0
    running_domain_loss = 0
    # 初始化总命中数、总目标数、总样本数
    total_hit_sou, total_num_sou, total_hit_tar, total_num_tar = 0.0, 0.0, 0.0, 0.0
    # 初始化测试准确率列表
    test_acc_list = []
    # 记录训练开始时间
    train_start = time.time()
    # 循环进行训练，共3001个episode
    for episode in range(5001):  # EPISODE = 90000
        # 获取源域少样本分类样本
        task_sou = utils.Task(metatrain_data, CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)  # 设置任务参数，如类别数、每类样本数等
        # 获取支持集的数据加载器
        support_dataloader_sou = utils.get_HBKC_data_loader(task_sou, num_per_class=SHOT_NUM_PER_CLASS, split="train",
                                                        shuffle=False)
        # 获取查询集的数据加载器
        query_dataloader_sou = utils.get_HBKC_data_loader(task_sou, num_per_class=QUERY_NUM_PER_CLASS, split="test",
                                                      shuffle=False)

        # 从支持集和查询集的数据加载器中获取数据
        supports_sou, support_labels_sou = support_dataloader_sou.__iter__().__next__()  # 获取支持集数据和标签
        querys_sou, query_labels_sou = query_dataloader_sou.__iter__().__next__()  # 获取查询集数据和标签

        # 获取源域支持集的真实标签
        query_real_labels_sou = task_sou.query_real_labels
        # print(query_real_labels_sou)

        # 初始化源域的语义支持特征为零张量
        semantic_query_sou = torch.zeros(CLASS_NUM*QUERY_NUM_PER_CLASS, 768)
        semantic_query_sou_all = torch.zeros(CLASS_NUM * QUERY_NUM_PER_CLASS, 768)
        # 根据源域支持集的真实标签，从语义映射中获取对应的语义特征

        for i, class_id in enumerate(query_real_labels_sou):
            # print(i,class_id)
            semantic_query_sou[i] = torch.from_numpy(semantic_mapping_sou[class_id])
            semantic_query_sou_all[i] = torch.from_numpy(semantic_mapping_sou_all)


        semantic_features_sou = semantic_feature_encoder(semantic_query_sou.cuda())
        semantic_features_sou_all = semantic_feature_encoder(semantic_query_sou_all.cuda())
        # print(semantic_features_sou.shape)
        # print(semantic_features_sou_all.shape)
        # 使用映射网络对支持集和查询集数据进行处理
        supports_sou = mapping_network(supports_sou.cuda(), domain='source')
        querys_sou = mapping_network(querys_sou.cuda(), domain='source')
        # semantic_features_sou = mapping_network(semantic_features_sou.cuda(), domain='source')


        # 获取目标域的少样本分类样本
        task_tar = utils.Task(target_da_metatrain_data, TEST_CLASS_NUM, SHOT_NUM_PER_CLASS,
                          QUERY_NUM_PER_CLASS)  # 定义任务参数
        support_dataloader_tar = utils.get_HBKC_data_loader(task_tar, num_per_class=SHOT_NUM_PER_CLASS, split="train",
                                                        shuffle=False)  # 获取支持集数据加载器
        query_dataloader_tar = utils.get_HBKC_data_loader(task_tar, num_per_class=QUERY_NUM_PER_CLASS, split="test",
                                                      shuffle=False)  # 获取查询集数据加载器

        # 从支持集和查询集的数据加载器中获取数据
        supports_tar, support_labels_tar = support_dataloader_tar.__iter__().__next__()  # 获取支持集数据和标签
        querys_tar, query_labels_tar = query_dataloader_tar.__iter__().__next__()  # 获取查询集数据和标签

        # 获取目标域支持集的真实标签
        query_real_labels_tar = task_tar.query_real_labels
        semantic_query_tar = torch.zeros(TEST_CLASS_NUM*QUERY_NUM_PER_CLASS, 768)
        semantic_query_tar_all = torch.zeros(TEST_CLASS_NUM * QUERY_NUM_PER_CLASS, 768)
        # 根据目标域支持集的真实标签，从语义映射中获取对应的语义特征
        for i, class_id in enumerate(query_real_labels_tar):
            semantic_query_tar[i] = torch.from_numpy(semantic_mapping_tar[class_id])
            semantic_query_tar_all[i] = torch.from_numpy(semantic_mapping_tar_all)

        semantic_features_tar = semantic_feature_encoder(semantic_query_tar.cuda())
        semantic_features_tar_all = semantic_feature_encoder(semantic_query_tar_all.cuda())
        # 使用映射网络对支持集和查询集数据进行处理
        supports_tar = mapping_network(supports_tar.cuda(), domain='target')  # 对支持集数据进行域适应处理
        querys_tar = mapping_network(querys_tar.cuda(), domain='target')  # 对查询集数据进行域适应处理
        # semantic_features_tar = mapping_network(semantic_features_tar.cuda(), domain='target')
        # 在前1000个episode中，进行源域的少样本分类和域适应

        # 计算特征
        support_features_sou, supp_att_sou = feature_encoder(supports_sou.cuda())  # 计算支持集的特征
        query_features_sou, query_att_sou = feature_encoder(querys_sou.cuda())  # 计算查询集的特征
        query_features_tar, query_att_tar = feature_encoder(querys_tar.cuda())  # 计算查询集的特征
        support_features_tar, supp_att_tar = feature_encoder(supports_tar.cuda())  # 计算支持集的特征

        # 使用CutMix对查询集数据进行数据增强
        mix_querys_1_sou, query_label_11_sou, query_label_12_sou, lam_11_sou, lam_12_sou = cutmix1(querys_sou, query_labels_sou, query_att_sou,alpha=1.0)
        mix_querys_2_sou, query_label_21_sou, query_label_22_sou, lam_21_sou, lam_22_sou = cutmix2(querys_sou, query_labels_sou, query_att_sou, querys_tar, query_labels_tar, query_att_tar,alpha=1.0)
        # 计算混合查询集的特征
        mix_query_features_1_sou, mix_query_outputs_1_sou = feature_encoder(mix_querys_1_sou.cuda())  # torch.Size([409, 32, 7, 3, 3])
        mix_query_features_2_sou, mix_query_outputs_2_sou = feature_encoder(mix_querys_2_sou.cuda())  # torch.Size([409, 32, 7, 3, 3])

        # print(lam_11_sou.shape)
        # print(lam_12_sou.shape)

        # 使用CutMix对查询集数据进行数据增强
        mix_querys_1_tar, query_label_11_tar, query_label_12_tar, lam_11_tar, lam_12_tar = cutmix1(querys_tar, query_labels_tar, query_att_tar,alpha=1.0)
        mix_querys_2_tar, query_label_21_tar, query_label_22_tar, lam_21_tar, lam_22_tar = cutmix2(querys_tar, query_labels_tar, query_att_tar, querys_sou, query_labels_sou, query_att_sou,alpha=1.0)
        # 计算混合查询集的特征
        mix_query_features_1_tar, mix_query_outputs_1_tar = feature_encoder(mix_querys_1_tar.cuda())  # torch.Size([409, 32, 7, 3, 3])
        mix_query_features_2_tar, mix_query_outputs_2_tar = feature_encoder(mix_querys_2_tar.cuda())  # torch.Size([409, 32, 7, 3, 3])

        # 原型网络
        if SHOT_NUM_PER_CLASS > 1:
            # 如果每类样本数大于1，则计算支持集的原型
            support_proto_sou = support_features_sou.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)  # (9, 160)
            support_proto_tar = support_features_tar.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)  # (9, 160)
        else:
            # 如果每类样本数为1，则直接使用支持集的特征作为原型
            support_proto_sou = support_features_sou
            support_proto_tar = support_features_tar

        # 计算少样本分类损失
        logit_sou = euclidean_metric(query_features_sou, support_proto_sou)
        logits_1_sou = euclidean_metric(mix_query_features_1_sou, support_proto_sou)  # 使用欧几里得距离计算分类结果
        logits_2_sou = euclidean_metric(mix_query_features_2_sou, support_proto_sou)  # 使用欧几里得距离计算分类结果

        f_loss_sou = F.cross_entropy(logit_sou, query_labels_sou.long().cuda(), reduction="none")
        f_loss_1_sou = lam_11_sou * F.cross_entropy(logits_1_sou, query_label_11_sou.long().cuda(), reduction="none") + lam_12_sou * F.cross_entropy(logits_1_sou, query_label_12_sou.long().cuda(), reduction="none")
        f_loss_2_sou = lam_21_sou * F.cross_entropy(logits_2_sou, query_label_21_sou.long().cuda(), reduction="none") + lam_22_sou * F.cross_entropy(logits_2_sou, query_label_22_sou.long().cuda(), reduction="none")

        f_loss_sou = torch.mean(f_loss_sou, axis=0)
        f_loss_1_sou = torch.mean(f_loss_1_sou, axis=0)  # 计算损失的平均值
        f_loss_2_sou = torch.mean(f_loss_1_sou, axis=0)  # 计算损失的平均值

        # 计算少样本分类损失
        logit_tar = euclidean_metric(query_features_tar, support_proto_tar)
        logits_1_tar = euclidean_metric(mix_query_features_1_tar, support_proto_tar)  # 使用欧几里得距离计算分类结果
        logits_2_tar = euclidean_metric(mix_query_features_2_tar, support_proto_tar)  # 使用欧几里得距离计算分类结果

        f_loss_tar = F.cross_entropy(logit_tar, query_labels_tar.long().cuda(), reduction="none")
        f_loss_1_tar = lam_11_tar * F.cross_entropy(logits_1_tar, query_label_11_tar.long().cuda(), reduction="none") + lam_12_tar * F.cross_entropy(logits_1_tar, query_label_12_tar.long().cuda(), reduction="none")
        f_loss_2_tar = lam_21_tar * F.cross_entropy(logits_2_tar, query_label_21_tar.long().cuda(), reduction="none") + lam_22_tar * F.cross_entropy(logits_2_tar, query_label_22_tar.long().cuda(), reduction="none")

        f_loss_tar = torch.mean(f_loss_tar, axis=0)
        f_loss_1_tar = torch.mean(f_loss_1_tar, axis=0)  # 计算损失的平均值
        f_loss_2_tar = torch.mean(f_loss_1_tar, axis=0)  # 计算损失的平均值

        # 显式扩展 lambda 张量的维度
        lam_11_sou = lam_11_sou.unsqueeze(1)  # [304] → [304, 1]
        lam_12_sou = lam_12_sou.unsqueeze(1)  # [304] → [304, 1]
        semantic_features_sou_mix = lam_11_sou*semantic_features_sou + lam_12_sou*semantic_features_sou_all

        lam_11_tar = lam_11_tar.unsqueeze(1)  # [304] → [304, 1]
        lam_12_tar = lam_12_tar.unsqueeze(1)  # [304] → [304, 1]
        semantic_features_tar_mix = lam_11_tar * semantic_features_tar + lam_12_tar * semantic_features_tar_all

        text_align_loss_sou = image_text_contrastive_loss(mix_query_features_1_sou.cuda(), semantic_features_sou_mix.cuda())
        text_align_loss_tar = image_text_contrastive_loss(mix_query_features_1_tar.cuda(), semantic_features_tar_mix.cuda())

        # joint_feature_sou = 0.5 * mix_query_features_1_sou + (1 - 0.5) * semantic_features_sou_mix
        # joint_feature_tar = 0.5 * mix_query_features_1_tar + (1 - 0.5) * semantic_features_tar_mix
        # joint_feature_sou = attn_layer(semantic_features_sou_mix.cuda() , mix_query_features_1_sou.cuda())
        # joint_feature_tar = attn_layer(semantic_features_tar_mix.cuda() , mix_query_features_1_tar.cuda())
        joint_feature_sou = attn_layer(mix_query_features_1_sou.cuda(),semantic_features_sou_mix.cuda())
        joint_feature_tar = attn_layer(mix_query_features_1_tar.cuda(),semantic_features_tar_mix.cuda())
        # print(mix_query_features_1_sou.shape)
        # print(semantic_features_sou_mix.shape)
        joint_logits_sou = euclidean_metric(joint_feature_sou, support_proto_sou)
        joint_logits_tar = euclidean_metric(joint_feature_tar, support_proto_tar)

        f_loss_joint_sou = F.cross_entropy(joint_logits_sou, query_labels_sou.long().cuda(), reduction="none")
        f_loss_joint_tar = F.cross_entropy(joint_logits_tar, query_labels_tar.long().cuda(), reduction="none")
        f_loss_joint_sou = torch.mean(f_loss_joint_sou, axis=0)
        f_loss_joint_tar = torch.mean(f_loss_joint_tar, axis=0)

        f_loss = f_loss_sou + f_loss_tar
        f_loss_1 = f_loss_1_sou + f_loss_1_tar
        f_loss_2 = f_loss_2_sou + f_loss_2_tar
        text_align_loss = text_align_loss_sou + text_align_loss_tar
        f_loss_joint = f_loss_joint_sou + f_loss_joint_tar
        # 总损失等于少样本分类损失
        loss = f_loss +2.5*(f_loss_1 +  f_loss_2 )+1 * (text_align_loss + f_loss_joint)

        # 更新模型参数
        feature_encoder.zero_grad()  # 清除特征编码器的梯度
        mapping_network.zero_grad()  # 清除映射网络的梯度
        semantic_feature_encoder.zero_grad()
        loss.backward()  # 反向传播，计算梯度
        feature_encoder_optim.step()  # 更新特征编码器的参数
        mapping_network_optim.step()
        semantic_feature_encoder_optim.step()

        # 计算训练过程中的准确率
        total_hit_sou += torch.sum(torch.argmax(logit_sou.cuda(), dim=1).cpu() == query_labels_sou).item()  # 累加命中次数
        total_num_sou += querys_sou.shape[0]  # 累加总样本数
        # total_hit_sou += torch.sum(torch.argmax(logits_1_sou.cuda(), dim=1).cpu() == query_labels_sou).item()  # 累加命中次数
        # total_hit_sou += torch.sum(torch.argmax(logits_2_sou.cuda(), dim=1).cpu() == query_labels_sou).item()  # 累加命中次数
        # total_num_sou += querys_sou.shape[0]  # 累加总样本数
        # total_num_sou += querys_sou.shape[0]  # 累加总样本数
        acc_sou = total_hit_sou / total_num_sou

        # 计算训练过程中的准确率
        total_hit_tar += torch.sum(torch.argmax(logit_tar.cuda(), dim=1).cpu() == query_labels_tar).item()  # 累加命中次数
        total_num_tar += querys_tar.shape[0]  # 累加总样本数
        # total_hit_tar += torch.sum(torch.argmax(logits_1_tar.cuda(), dim=1).cpu() == query_labels_tar).item()  # 累加命中次数
        # total_hit_tar += torch.sum(torch.argmax(logits_2_tar.cuda(), dim=1).cpu() == query_labels_tar).item()  # 累加命中次数
        # total_num_tar += querys_tar.shape[0]  # 累加总样本数
        # total_num_tar += querys_tar.shape[0]  # 累加总样本数
        acc_tar = total_hit_tar / total_num_tar

        # 每隔100个episode打印训练信息（源域）
        if (episode + 1) % 100 == 0 :
            train_loss.append(loss.item())  # 记录训练损失
            print('episode {:>3d}:  acc_sou: {:6.4f}, acc_tar: {:6.4f}, loss: {:6.4f}'.format(episode + 1, acc_sou, acc_tar, loss.item()))

#         # ✅ 替换原始的测试模块（if (episode + 1) % 500 == 0 or episode == 0）为以下完整段落,生成t-SNE图
#         # ✅ 替换测试部分，添加 t-SNE 特征保存逻辑
#         if (episode + 1) % 200 == 1 or episode == 0:
#             with torch.no_grad():
#
#                 print("Testing ...")
#                 train_end = time.time()
#                 mapping_network.eval()
#                 feature_encoder.eval()
#                 semantic_feature_encoder.eval()
#
#                 total_rewards = 0
#                 counter = 0
#                 accuracies = []
#                 predict = np.array([], dtype=np.int64)
#                 labels = np.array([], dtype=np.int64)
#
#                 train_datas, train_labels = train_loader.__iter__().__next__()
#                 train_datas = mapping_network(Variable(train_datas).cuda(), domain='target')
#                 train_features, _ = feature_encoder(train_datas)
#
#                 max_value = train_features.max()
#                 min_value = train_features.min()
#                 train_features = (train_features - min_value) * 1.0 / (max_value - min_value)
#
#                 KNN_classifier = KNeighborsClassifier(n_neighbors=1, weights='distance')
#                 KNN_classifier.fit(train_features.cpu().detach().numpy(), train_labels)
#
#                 all_features = []
#                 all_labels = []
#
#                 for test_datas, test_labels in test_loader:
#                     batch_size = test_labels.shape[0]
#                     test_datas = mapping_network(Variable(test_datas).cuda(), domain='target')
#                     test_features, _ = feature_encoder(test_datas)
#                     test_features = (test_features - min_value) * 1.0 / (max_value - min_value)
#                     test_features = test_features.cpu()
#
#                     predict_labels = KNN_classifier.predict(test_features.detach().numpy())
#                     test_labels_np = test_labels.numpy()
#
#                     all_features.append(test_features.numpy())
#                     all_labels.append(test_labels_np)
#
#                     rewards = [1 if predict_labels[j] == test_labels_np[j] else 0 for j in range(batch_size)]
#                     total_rewards += np.sum(rewards)
#                     counter += batch_size
#
#                     predict = np.append(predict, predict_labels)
#                     labels = np.append(labels, test_labels_np)
#
#                     accuracy = total_rewards / 1.0 / counter
#                     accuracies.append(accuracy)
#
#                 test_accuracy = 100. * total_rewards / len(test_loader.dataset)
#                 print('\t\tepisode {} accuracy: {}/{} ({:.2f}%)\n'.format(
#                     episode + 1, total_rewards, len(test_loader.dataset), test_accuracy))
#
#                 test_end = time.time()
#                 mapping_network.train()
#                 feature_encoder.train()
#                 semantic_feature_encoder.train()
#
#                 if test_accuracy > last_accuracy:
#                     torch.save(mapping_network.state_dict(),
#                                f"./checkpoints/DFSL_mapping_network_IP_{iDataSet}iter_{TEST_LSAMPLE_NUM_PER_CLASS}shot.pkl")
#                     torch.save(feature_encoder.state_dict(),
#                                f"./checkpoints/DFSL_feature_encoder_IP_{iDataSet}iter_{TEST_LSAMPLE_NUM_PER_CLASS}shot.pkl")
#                     print("save networks for episode:", episode + 1)
#
#                     last_accuracy = test_accuracy
#                     best_episdoe = episode
#
#                     acc[iDataSet] = test_accuracy
#                     OA = acc
#                     C = metrics.confusion_matrix(labels, predict)
#                     A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float64)
#                     k[iDataSet] = metrics.cohen_kappa_score(labels, predict)
#
#                     # ✅ 保存最佳特征用于 t-SNE
#                     best_tsne_features = np.concatenate(all_features, axis=0)
#                     best_tsne_labels = np.concatenate(all_labels, axis=0)
#
#                 print('best episode:[{}], best accuracy={}'.format(best_episdoe + 1, last_accuracy))
#
# # ✅ 插入到每个 iDataSet 训练结束后（即 logger.info('*****') 前）
# import scipy.io as sio
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from sklearn.preprocessing import StandardScaler
# from sklearn.utils import shuffle
# label_colors = {
#     1: [0, 0, 1],
#     2: [1.000, 0.498, 0.055],
#     3: [0, 1, 1],
#     4: [1, 0, 0],
#     5: [1, 0, 1],
#     6: [1, 1, 0],
#     7: [0.5, 0.5, 1],
#     8: [0.65, 0.35, 1],
#     9: [0.75, 0.5, 0.75],
#     10: [0.1, 0.5, 0.6],
#     11: [0.9, 1, 0.65],
#     12: [0.65, 0.65, 0],
#     13: [0.1, 0.7, 0.1],
#     14: [0.392, 0.584, 0.929],
#     15: [0.690, 0.314, 0.125],
#     16: [0.5, 0.75, 1]
# }
# if best_tsne_features is not None and best_tsne_labels is not None:
#     print('Generating t-SNE visualization of best round...')
#     features_std = StandardScaler().fit_transform(best_tsne_features)
#     features_std, best_tsne_labels = shuffle(features_std, best_tsne_labels, random_state=0)
#     features_std = features_std[:10000]
#     best_tsne_labels = best_tsne_labels[:10000]
#     tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=0)
#     features_tsne = tsne.fit_transform(features_std)
#
#     plt.figure(figsize=(10, 8))
#     for label in np.unique(best_tsne_labels):
#         idx = best_tsne_labels == label
#         color = label_colors.get(label+1, [0.5, 0.5, 0.5])
#         plt.scatter(features_tsne[idx, 0], features_tsne[idx, 1], color=color,s=25, label=f'{label+1}')
#
#     # ----✨ 设置坐标轴字体大小（可调）✨----
#     axis_fontsize = 14  # 🔧 这里改这个值即可控制字号
#     plt.tick_params(axis='both', which='major', labelsize=25)
#
#     plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=20)
#     plt.tight_layout()
#     plt.show()
#
#     plt.savefig('./OUR-UP3-T-SNE_plot.png')  # 保存图形为文件

        # 每隔200个episode或在第一个episode时进行测试
        if (episode + 1) % 200 == 1 or episode == 0:
            with torch.no_grad():

                print("Testing ...")  # 提示开始测试
                train_end = time.time()  # 记录训练结束时间
                mapping_network.eval()  # 将映射网络设置为评估模式
                feature_encoder.eval()  # 将特征编码器设置为评估模式
                semantic_feature_encoder.eval()

                total_rewards = 0  # 初始化总奖励
                counter = 0  # 初始化样本计数器
                accuracies = []  # 初始化准确率列表
                predict = np.array([], dtype=np.int64)  # 初始化预测结果数组
                labels = np.array([], dtype=np.int64)  # 初始化真实标签数组

                # 获取训练数据并进行预处理
                train_datas, train_labels = train_loader.__iter__().__next__()
                train_datas = mapping_network(Variable(train_datas).cuda(), domain='target')  # 对训练数据进行域适应处理
                train_features, _ = feature_encoder(train_datas)  # 提取训练数据的特征

                # 归一化特征
                max_value = train_features.max()  # 获取特征的最大值
                min_value = train_features.min()  # 获取特征的最小值
                # print(max_value.item())  # 打印最大值
                # print(min_value.item())  # 打印最小值
                train_features = (train_features - min_value) * 1.0 / (max_value - min_value)  # 归一化特征

                # 使用KNN分类器进行分类
                KNN_classifier = KNeighborsClassifier(n_neighbors=1, weights='distance')  # 初始化KNN分类器
                KNN_classifier.fit(train_features.cpu().detach().numpy(), train_labels)  # 使用训练特征和标签训练KNN分类器

                # 对测试数据进行评估
                for test_datas, test_labels in test_loader:
                    batch_size = test_labels.shape[0]  # 获取当前批次的样本数

                    # 对测试数据进行预处理
                    test_datas = mapping_network(Variable(test_datas).cuda(), domain='target')  # 对测试数据进行域适应处理
                    test_features, _ = feature_encoder(test_datas)  # 提取测试数据的特征
                    test_features = (test_features - min_value) * 1.0 / (max_value - min_value)  # 归一化特征

                    # 使用KNN分类器进行预测
                    predict_labels = KNN_classifier.predict(test_features.cpu().detach().numpy())
                    test_labels = test_labels.numpy()  # 将真实标签转换为numpy数组

                    # 计算准确率
                    rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(batch_size)]  # 计算每个样本是否预测正确
                    total_rewards += np.sum(rewards)  # 累加总奖励
                    counter += batch_size  # 累加样本计数器

                    # 累加预测结果和真实标签
                    predict = np.append(predict, predict_labels)
                    labels = np.append(labels, test_labels)

                    # 计算当前批次的准确率
                    accuracy = total_rewards / 1.0 / counter
                    accuracies.append(accuracy)

                # 计算测试准确率
                test_accuracy = 100. * total_rewards / len(test_loader.dataset)
                print('\t\tepisode {} accuracy: {}/{} ({:.2f}%)\n'.format(episode + 1,total_rewards, len(test_loader.dataset),
                                                               100. * total_rewards / len(test_loader.dataset)))
                test_end = time.time()  # 记录测试结束时间

                # 将模型切换到训练模式
                mapping_network.train()
                feature_encoder.train()
                semantic_feature_encoder.train()

                # 如果当前测试准确率高于上次记录的准确率，则保存模型
                if test_accuracy > last_accuracy:
                    # 保存映射网络和特征编码器的参数
                    torch.save(mapping_network.state_dict(),
                               str("./checkpoints/DFSL_mapping_network_" + "IP_" + str(iDataSet) + "iter_" + str(
                                   TEST_LSAMPLE_NUM_PER_CLASS) + "shot.pkl"))
                    torch.save(feature_encoder.state_dict(),
                               str("./checkpoints/DFSL_feature_encoder_" + "IP_" + str(iDataSet) + "iter_" + str(
                                   TEST_LSAMPLE_NUM_PER_CLASS) + "shot.pkl"))
                    print("save networks for episode:", episode + 1)  # 提示保存模型
                    last_accuracy = test_accuracy  # 更新记录的最高准确率
                    best_episdoe = episode  # 更新最佳episode

                    # 更新当前实验的准确率和混淆矩阵
                    acc[iDataSet] = 100. * total_rewards / len(test_loader.dataset)  # 计算准确率
                    OA = acc  # 总体准确率（Overall Accuracy）
                    C = metrics.confusion_matrix(labels, predict)  # 计算混淆矩阵
                    A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float64)  # 计算每类的准确率
                    k[iDataSet] = metrics.cohen_kappa_score(labels, predict)# 计算Cohen's Kappa

                # 打印当前最佳episode和准确率
                print('best episode:[{}], best accuracy={}'.format(best_episdoe + 1, last_accuracy))
    # 如果当前测试准确率高于全局最高准确率，则更新全局最佳预测结果和相关参数
    if test_accuracy > best_acc_all:
        best_predict_all = predict# 更新全局最佳预测结果
        best_G, best_RandPerm, best_Row, best_Column, best_nTrain = G, RandPerm, Row, Column, nTrain # 更新全局最佳参数
        # 打印当前实验的统计信息
    print('iter:{} best episode:[{}], best accuracy={}'.format(iDataSet, best_episdoe + 1, last_accuracy))
    print('***********************************************************************************')

# 计算每个实验的平均准确率（AA）和标准差
AA = np.mean(A, 1)  # 每个实验的平均准确率
AAMean = np.mean(AA, 0)  # 所有实验的平均准确率
AAStd = np.std(AA)  # 所有实验的准确率标准差

# 计算每个类别的平均准确率和标准差
AMean = np.mean(A, 0)  # 每个类别的平均准确率
AStd = np.std(A, 0)  # 每个类别的准确率标准差

# 计算总体准确率（OA）的平均值和标准差
OAMean = np.mean(acc)
OAStd = np.std(acc)

kMean = np.mean(k)
kStd = np.std(k)
# 打印训练和测试时间
print("-------------------------------")
print("train time per DataSet(s): " + "{:.5f}".format(train_end - train_start))
print("test time per DataSet(s): " + "{:.5f}".format(test_end - train_end))

print("-------------------------------")
# 打印总体准确率（OA）、平均准确率（AA）和Cohen's Kappa的统计信息
print("average OA: " + "{:.2f}".format(OAMean) + " +- " + "{:.2f}".format(OAStd))
print("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
print("average kappa: " + "{:.4f}".format(100 * kMean) + " +- " + "{:.4f}".format(100 * kStd))

# 打印每个类别的准确率统计信息
print("-------------------------------")
print("accuracy for each class: ")
for i in range(CLASS_NUM):
    print("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))

# 找到所有实验中准确率最高的实验
print("-------------------------------")
best_iDataset = 0
for i in range(len(acc)):
    print('{}:{}'.format(i, acc[i]))
    if acc[i] > acc[best_iDataset]:
        best_iDataset = i
print('best acc all={}:{}'.format(i,acc[best_iDataset]))

# #################classification map################################
#
# # 根据最佳预测结果生成分类图
# for i in range(len(best_predict_all)):  # 遍历所有预测结果
#     best_G[best_Row[best_RandPerm[best_nTrain + i]]][best_Column[best_RandPerm[best_nTrain + i]]] = \
#     best_predict_all[i] + 1  # 将预测结果映射到原始图像位置
#
# # 创建用于显示的分类图
# hsi_pic = np.zeros((best_G.shape[0], best_G.shape[1], 3))  # 初始化分类图
# for i in range(best_G.shape[0]):
#     for j in range(best_G.shape[1]):
#         if best_G[i][j] == 0:
#             hsi_pic[i, j, :] = [0, 0, 0]  # 背景
#         if best_G[i][j] == 1:
#             hsi_pic[i, j, :] = [0, 0, 1]  # 类别1
#         if best_G[i][j] == 2:
#             hsi_pic[i, j, :] = [0, 1, 0]  # 类别2
#         if best_G[i][j] == 3:
#             hsi_pic[i, j, :] = [0, 1, 1]  # 类别3
#         if best_G[i][j] == 4:
#             hsi_pic[i, j, :] = [1, 0, 0]  # 类别4
#         if best_G[i][j] == 5:
#             hsi_pic[i, j, :] = [1, 0, 1]  # 类别5
#         if best_G[i][j] == 6:
#             hsi_pic[i, j, :] = [1, 1, 0]  # 类别6
#         if best_G[i][j] == 7:
#             hsi_pic[i, j, :] = [0.5, 0.5, 1]  # 类别7
#         if best_G[i][j] == 8:
#             hsi_pic[i, j, :] = [0.65, 0.35, 1]  # 类别8
#         if best_G[i][j] == 9:
#             hsi_pic[i, j, :] = [0.75, 0.5, 0.75]  # 类别9
#
# # 保存分类图
# utils.classification_map(hsi_pic[4:-4, 4:-4, :], best_G[4:-4, 4:-4], 24, "./F3UP_{}shot.png".format(TEST_LSAMPLE_NUM_PER_CLASS))