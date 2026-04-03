import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

import numpy as np
import os
import math
import argparse
import scipy as sp
import scipy.stats
import pickle
import random
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot
from matplotlib.colors import ListedColormap
import time
import utils
import sys
from torchvision import transforms
from vit import Transformer
import vit
from thop import profile
from ptflops import get_model_complexity_info


parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")


parser.add_argument("-f", "--feature_dim", type=int, default=160, help="Dimension of the feature space")
parser.add_argument("-c", "--src_input_dim", type=int, default=128, help="Dimension of the input data in the source domain")
parser.add_argument("-d", "--tar_input_dim", type=int, default=200, help="Dimension of the input data in the target domain")
parser.add_argument("-n", "--n_dim", type=int, default=100, help="Dimension of the N space")
parser.add_argument("-w", "--class_num", type=int, default=16, help="Number of classes")
parser.add_argument("-s", "--shot_num_per_class", type=int, default=1, help="Number of support samples per class")
parser.add_argument("-b", "--query_num_per_class", type=int, default=19, help="Number of query samples per class")
parser.add_argument("-e", "--episode", type=int, default=20000, help="Number of training episodes")
parser.add_argument("-t", "--test_episode", type=int, default=600, help="Number of testing episodes")
parser.add_argument("-l", "--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
parser.add_argument("-g", "--gpu", type=int, default=4, help="GPU device number to use")
parser.add_argument("-u", "--hidden_unit", type=int, default=10, help="Number of hidden units in the network")

import argparse
import random
import os
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--test_class_num", type=int, default=16)
parser.add_argument("-z", "--test_lsample_num_per_class", type=int, default=5)
parser.add_argument("-a", "--trans_layer", type=int, default=2)

args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

FEATURE_DIM = args.feature_dim
SRC_INPUT_DIMENSION = args.src_input_dim
TAR_INPUT_DIMENSION = args.tar_input_dim
N_DIMENSION = args.n_dim
CLASS_NUM = args.class_num
SHOT_NUM_PER_CLASS = args.shot_num_per_class
QUERY_NUM_PER_CLASS = args.query_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit

TEST_CLASS_NUM = args.test_class_num
TEST_LSAMPLE_NUM_PER_CLASS = args.test_lsample_num_per_class

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(0)


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
#IP
labels_tar = [
    "Alfalfa: A type of flowering plant in the pea family, used as forage for livestock.",
    "Corn notill: Maize cultivated with zero-tillage techniques, leaving crop residue on the soil surface.",
    "Corn mintill: Corn planted using minimum tillage, disturbing the soil minimally to prepare for planting.",
    "Corn: A general term for maize, a major cereal crop used for food, animal feed, and industrial products.",
    "Grass pasture: Areas covered with grasses used for grazing livestock.",
    "Grass trees: Areas with scattered trees and grass, typical of certain savanna ecosystems.",
    "Grass pasture mowed: Pasture land that has been mowed, often to control growth or prepare for grazing.",
    "Hay windrowed: Hay that has been cut and left in rows (windrows) to dry before being baled.",
    "Oats: A cereal grain grown for its seed, used for animal feed and human food.",
    "Soybean notill: Legumes seeded with soil-conserving practices, minimizing soil disturbance.",
    "Soybean mintill: Soybeans planted with minimal tillage, slightly disturbing the soil.",
    "Soybean clean: Soybean fields without visible crop residue, likely post-harvest.",
    "Wheat: A cereal grain widely cultivated for its seed, a staple food in many countries.",
    "Woods:  Fields dominated by trees and shrubs, forming a woodland or forest.",
    "Buildings Grass Trees Drives: Areas with a mix of buildings, grassy areas, trees, and roads.",
    "Stone Steel Towers: Structures made of stone or steel, often for industrial or communication purposes."
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
#IP
labels_tar_all = [
    "Alfalfa: A type of flowering plant in the pea family, used as forage for livestock,\
    Corn notill: Maize cultivated with zero-tillage techniques, leaving crop residue on the soil surface,\
    Corn mintill: Corn planted using minimum tillage, disturbing the soil minimally to prepare for planting,\
    Corn: A general term for maize, a major cereal crop used for food, animal feed, and industrial products,\
    Grass pasture: Areas covered with grasses used for grazing livestock,\
    Grass trees: Areas with scattered trees and grass, typical of certain savanna ecosystems,\
    Grass pasture mowed: Pasture land that has been mowed, often to control growth or prepare for grazing,\
    Hay windrowed: Hay that has been cut and left in rows (windrows) to dry before being baled,\
    Oats: A cereal grain grown for its seed, used for animal feed and human food,\
    Soybean notill: Legumes seeded with soil-conserving practices, minimizing soil disturbance,\
    Soybean mintill: Soybeans planted with minimal tillage, slightly disturbing the soil,\
    Soybean clean: Soybean fields without visible crop residue, likely post-harvest,\
    Wheat: A cereal grain widely cultivated for its seed, a staple food in many countries,\
    Woods:  Fields dominated by trees and shrubs, forming a woodland or forest,\
    Buildings Grass Trees Drives: Areas with a mix of buildings, grassy areas, trees, and roads,\
    Stone Steel Towers: Structures made of stone or steel, often for industrial or communication purposes."
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


def _init_():
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    if not os.path.exists('./classificationMap'):
        os.makedirs('./classificationMap')

_init_()

with open(os.path.join('../datasets', '/data/yuchao/data/yuchao/Datasets/Chikusei_imdb_128.pickle'), 'rb') as handle:
    source_imdb = pickle.load(handle)
print(source_imdb.keys())
print(source_imdb['Labels'])

data_train = source_imdb['data']
labels_train = source_imdb['Labels']
print(data_train.shape)
print(labels_train.shape)
keys_all_train = sorted(list(set(labels_train)))
print(keys_all_train)
label_encoder_train = {}
for i in range(len(keys_all_train)):
    label_encoder_train[keys_all_train[i]] = i
print(label_encoder_train)

train_set = {}
for class_, path in zip(labels_train, data_train):
    if label_encoder_train[class_] not in train_set:
        train_set[label_encoder_train[class_]] = []
    train_set[label_encoder_train[class_]].append(path)
print(train_set.keys())
data = train_set
del train_set
del keys_all_train
del label_encoder_train

print("Num classes for source domain datasets: " + str(len(data)))
print(data.keys())
data = utils.sanity_check(data)
print("Num classes of the number of class larger than 200: " + str(len(data)))

for class_ in data:
    for i in range(len(data[class_])):
        image_transpose = np.transpose(data[class_][i], (2, 0, 1))
        data[class_][i] = image_transpose

metatrain_data = data
print(len(metatrain_data.keys()), metatrain_data.keys())
del data

print(source_imdb['data'].shape)
source_imdb['data'] = source_imdb['data'].transpose((1, 2, 3, 0))
print(source_imdb['data'].shape)
print(source_imdb['Labels'])
source_dataset = utils.matcifar(source_imdb, train=True, d=3, medicinal=0)
source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=128, shuffle=True, num_workers=2,pin_memory=True)
del source_dataset, source_imdb

test_data = '/data/yuchao/data/yuchao/Datasets/IP/indian_pines_corrected.mat'
test_label = '/data/yuchao/data/yuchao/Datasets/IP/indian_pines_gt.mat'

Data_Band_Scaler, GroundTruth = utils.load_data(test_data, test_label)

def get_train_test_loader(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    print(Data_Band_Scaler.shape)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape

    num_class = int(np.max(GroundTruth))
    data_band_scaler = utils.flip(Data_Band_Scaler)
    groundtruth = utils.flip(GroundTruth)
    del Data_Band_Scaler
    del GroundTruth

    HalfWidth = 4
    G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth]
    data = data_band_scaler[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth, :]

    [Row, Column] = np.nonzero(G)
    del data_band_scaler
    del groundtruth

    nSample = np.size(Row)
    print('number of sample', nSample)

    train = {}
    test = {}
    da_train = {}
    m = int(np.max(G))
    nlabeled = TEST_LSAMPLE_NUM_PER_CLASS
    print('labeled number per class:', nlabeled)
    print((200 - nlabeled) / nlabeled + 1)
    print(math.ceil((200 - nlabeled) / nlabeled) + 1)

    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if G[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        nb_val = shot_num_per_class
        train[i] = indices[:nb_val]
        da_train[i] = []
        for j in range(math.ceil((200 - nlabeled) / nlabeled) + 1):
            da_train[i] += indices[:nb_val]
        test[i] = indices[nb_val:]

    train_indices = []
    test_indices = []
    da_train_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
        da_train_indices += da_train[i]
    np.random.shuffle(test_indices)

    print('the number of train_indices:', len(train_indices))
    print('the number of test_indices:', len(test_indices))
    print('the number of train_indices after data argumentation:', len(da_train_indices))
    print('labeled sample indices:', train_indices)

    nTrain = len(train_indices)
    nTest = len(test_indices)
    da_nTrain = len(da_train_indices)

    imdb = {}
    imdb['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nTrain + nTest], dtype=np.float32)
    imdb['Labels'] = np.zeros([nTrain + nTest], dtype=np.int64)
    imdb['set'] = np.zeros([nTrain + nTest], dtype=np.int64)

    RandPerm = train_indices + test_indices
    RandPerm = np.array(RandPerm)

    for iSample in range(nTrain + nTest):
        imdb['data'][:, :, :, iSample] = data[Row[RandPerm[iSample]] - HalfWidth:  Row[RandPerm[iSample]] + HalfWidth + 1, Column[RandPerm[iSample]] - HalfWidth: Column[RandPerm[iSample]] + HalfWidth + 1, :]
        imdb['Labels'][iSample] = G[Row[RandPerm[iSample]], Column[RandPerm[iSample]]].astype(np.int64)

    imdb['Labels'] = imdb['Labels'] - 1
    imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nTest]))).astype(np.int64)
    print('Data is OK.')

    train_dataset = utils.matcifar(imdb, train=True, d=3, medicinal=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=class_num * shot_num_per_class, shuffle=False, num_workers=2,pin_memory=True)
    del train_dataset

    test_dataset = utils.matcifar(imdb, train=False, d=3, medicinal=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2,pin_memory=True)
    del test_dataset
    del imdb

    imdb_da_train = {}
    imdb_da_train['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, da_nTrain], dtype=np.float32)
    imdb_da_train['Labels'] = np.zeros([da_nTrain], dtype=np.int64)
    imdb_da_train['set'] = np.zeros([da_nTrain], dtype=np.int64)

    da_RandPerm = np.array(da_train_indices)

    for iSample in range(da_nTrain):
        imdb_da_train['data'][:, :, :, iSample] = utils.Crop_and_resize(data[Row[da_RandPerm[iSample]] - HalfWidth:  Row[da_RandPerm[iSample]] + HalfWidth + 1, Column[da_RandPerm[iSample]] - HalfWidth: Column[da_RandPerm[iSample]] + HalfWidth + 1, :])
        imdb_da_train['Labels'][iSample] = G[Row[da_RandPerm[iSample]], Column[da_RandPerm[iSample]]].astype(np.int64)

    imdb_da_train['Labels'] = imdb_da_train['Labels'] - 1
    imdb_da_train['set'] = np.ones([da_nTrain]).astype(np.int64)
    print('ok')

    return train_loader, test_loader, imdb_da_train, G, RandPerm, Row, Column, nTrain


def get_target_dataset(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    train_loader, test_loader, imdb_da_train, G, RandPerm, Row, Column, nTrain = get_train_test_loader(Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth, class_num=class_num, shot_num_per_class=shot_num_per_class)

    train_datas, train_labels = train_loader.__iter__().__next__()
    print('train labels:', train_labels)
    print('size of train datas:', train_datas.shape)

    print(imdb_da_train.keys())
    print(imdb_da_train['data'].shape)
    print(imdb_da_train['Labels'])

    del Data_Band_Scaler, GroundTruth

    target_da_datas = np.transpose(imdb_da_train['data'], (3, 2, 0, 1))
    print(target_da_datas.shape)
    target_da_labels = imdb_da_train['Labels']
    print('target data augmentation label:', target_da_labels)

    target_da_train_set = {}
    for class_, path in zip(target_da_labels, target_da_datas):
        if class_ not in target_da_train_set:
            target_da_train_set[class_] = []
        target_da_train_set[class_].append(path)
    target_da_metatrain_data = target_da_train_set
    print(target_da_metatrain_data.keys())

    target_dataset = utils.matcifar(imdb_da_train, train=True, d=3, medicinal=0)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=100, shuffle=True, num_workers=2,pin_memory=True)
    del target_dataset

    return train_loader, test_loader, target_da_metatrain_data, target_loader, G, RandPerm, Row, Column, nTrain

def conv3x3x3(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm3d(out_channel),
    )
    return layer

class Mapping(nn.Module):
    def __init__(self, in_dimension, out_dimension):
        super(Mapping, self).__init__()
        self.preconv = nn.Conv2d(in_dimension, out_dimension, kernel_size=1, stride=1, bias=False)
        self.preconv_bn = nn.BatchNorm2d(out_dimension)

    def forward(self, x):
        x = self.preconv(x)
        x = self.preconv_bn(x)
        return x

class Mapping_Network(nn.Module):
    def __init__(self):
        super(Mapping_Network, self).__init__()
        self.target_mapping = Mapping(TAR_INPUT_DIMENSION, N_DIMENSION)
        self.source_mapping = Mapping(SRC_INPUT_DIMENSION, N_DIMENSION)
        self.channel_shuffle = ChannelShuffle(10)
    def forward(self, x, domain='source'):
        if domain == 'target':
            x = self.target_mapping(x)
        elif domain == 'source':
            x = self.source_mapping(x)
        x = self.channel_shuffle(x)
        return x

class ChannelShuffle(nn.Module):
    def __init__(self, group):
        super(ChannelShuffle, self).__init__()
        self.group = group

    def forward(self, x):
        B, C, H, W = x.shape
        assert C % self.group == 0
        group_C = C // self.group
        x = x.view(B, self.group, group_C, H, W)
        x = x.transpose(1, 2).contiguous()
        x = x.view(B, C, H, W)
        return x

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.final_feat_dim = FEATURE_DIM
        self.cls_token = nn.Parameter(torch.randn(1, 1, N_DIMENSION))
        self.drop_x = nn.Dropout(0.5)
        self.transformer = Transformer(N_DIMENSION, args.trans_layer, 8, 64, 1024, 0.1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).transpose(1, 2)
        cls_tokens = self.cls_token.repeat(x.shape[0], 1, 1)
        x = torch.cat((x, cls_tokens), dim=1)
        x, att = self.transformer(x)
        x = x[:, -1, :]
        return x, att

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data = torch.ones(m.bias.data.size())

crossEntropy = nn.CrossEntropyLoss().cuda()
image_text_contrastive_loss = utils.ImageTextContrastiveLoss(batch_size=TEST_CLASS_NUM*19).cuda()

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b) ** 2).sum(dim=2)
    return logits

def cutmix1(data, target, querys_attention, alpha=1.0):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    querys_attention = querys_attention[:, -1, 0:81]
    querys_attention = querys_attention.squeeze(dim=1).view(-1, 9, 9)

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)

    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)

    atten_part = querys_attention[:, bbx1:bbx2, bby1:bby2]
    atten = torch.sum(querys_attention, axis=1)
    atten = torch.sum(atten, axis=1)
    atten_part = torch.sum(atten_part, axis=1)
    atten_part = torch.sum(atten_part, axis=1)
    lam = 1 - atten_part / atten

    shuffled_attention = querys_attention[indices]
    atten_part_b = shuffled_attention[:, bbx1:bbx2, bby1:bby2]
    atten_b = torch.sum(shuffled_attention, axis=1)
    atten_b = torch.sum(atten_b, axis=1)
    atten_part_b = torch.sum(atten_part_b, axis=1)
    atten_part_b = torch.sum(atten_part_b, axis=1)
    lam_b = atten_part_b / atten_b

    data[:, :, bbx1:bbx2, bby1:bby2] = shuffled_data[:, :, bbx1:bbx2, bby1:bby2]

    target_a = target
    target_b = shuffled_target

    return data, target_a, target_b, lam.cuda(), lam_b.cuda()

def cutmix2(data_a, target_a,querys_attention_a,data_b, target_b, querys_attention_b, alpha=1.0):
    indices = torch.randperm(data_b.size(0))
    shuffled_data_b = data_b[indices]
    shuffled_target_b = target_b[indices]

    querys_attention_a = querys_attention_a[:, -1, 0:81]
    querys_attention_a = querys_attention_a.squeeze(dim=1).view(-1, 9, 9)

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)

    bbx1, bby1, bbx2, bby2 = rand_bbox(data_b.size(), lam)

    atten_part_a = querys_attention_a[:, bbx1:bbx2, bby1:bby2]
    atten_a = torch.sum(querys_attention_a, axis=1)
    atten_a = torch.sum(atten_a, axis=1)
    atten_part_a = torch.sum(atten_part_a, axis=1)
    atten_part_a = torch.sum(atten_part_a, axis=1)
    lam_a = 1 - atten_part_a / atten_a

    shuffled_attention = querys_attention_b[indices]
    atten_part_b = shuffled_attention[:, bbx1:bbx2, bby1:bby2]
    atten_b = torch.sum(shuffled_attention, axis=1)
    atten_b = torch.sum(atten_b, axis=1)
    atten_part_b = torch.sum(atten_part_b, axis=1)
    atten_part_b = torch.sum(atten_part_b, axis=1)
    lam_b = atten_part_b / atten_b

    data_a[:, :, bbx1:bbx2, bby1:bby2] = shuffled_data_b[:, :, bbx1:bbx2, bby1:bby2]

    target_a = target_a
    target_b = shuffled_target_b

    return data_a, target_a, target_b, lam_a.cuda(), lam_b.cuda()


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class CrossModalAttention(nn.Module):
    def __init__(self, feat_dim=100, attn_dim=100, dropout=0.1):
        super().__init__()
        self.attn_dim = attn_dim
        self.W_q = nn.Linear(feat_dim, attn_dim).cuda()
        self.W_k = nn.Linear(feat_dim, attn_dim).cuda()
        self.W_v = nn.Linear(feat_dim, attn_dim).cuda()
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.zeros_(self.W_q.bias)
        nn.init.zeros_(self.W_k.bias)
        nn.init.zeros_(self.W_v.bias)

    def forward(self, text_feat, image_feat):
        Q = self.W_q(text_feat).cuda()
        K = self.W_k(image_feat).cuda()
        V = self.W_v(image_feat).cuda()

        attn_scores = torch.bmm(Q.unsqueeze(1), K.unsqueeze(2))
        attn_scores = attn_scores / (self.attn_dim ** 0.5)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        fused = attn_weights * V.unsqueeze(1)
        return fused.squeeze(1)

nDataSet = 10
acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, CLASS_NUM])
k = np.zeros([nDataSet, 1])
best_predict_all = []
best_acc_all = 0.0
best_G, best_RandPerm, best_Row, best_Column, best_nTrain = None, None, None, None, None
proto_loss = torch.zeros(1)
all_features = []
all_preds = []
seeds = [1272, 1018, 1308, 1309, 1084, 1490, 1499, 1114, 1123, 1194]

for iDataSet in range(nDataSet):
    print('---------Start of round {} ---------'.format(iDataSet))
    seed_torch(seeds[iDataSet])
    train_loader, test_loader, target_da_metatrain_data, target_loader, G, RandPerm, Row, Column, nTrain = get_target_dataset(
        Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth, class_num=TEST_CLASS_NUM, shot_num_per_class=TEST_LSAMPLE_NUM_PER_CLASS)
    feature_encoder = Network()

    semantic_feature_encoder = vit.WordEmbTransformers(feature_dim=100, dropout=0.5)
    mapping_network = Mapping_Network()
    feature_encoder.apply(weights_init)
    mapping_network.apply(weights_init)
    semantic_feature_encoder.apply(weights_init)
    feature_encoder.cuda()
    mapping_network.cuda()
    semantic_feature_encoder.cuda()
    feature_encoder.train()
    mapping_network.train()
    semantic_feature_encoder.train()
    feature_encoder_optim = torch.optim.Adam([{'params': feature_encoder.parameters()}], lr=args.learning_rate)
    mapping_network_optim = torch.optim.Adam([{'params':  mapping_network.parameters()}], lr=args.learning_rate)
    semantic_feature_encoder_optim = torch.optim.Adam([{'params':  semantic_feature_encoder.parameters()}], lr=args.learning_rate)
    attn_layer = CrossModalAttention()

    print("Training...")

    last_accuracy = 0.0
    best_episdoe = 0
    train_loss = []
    test_acc = []
    running_D_loss, running_F_loss = 0.0, 0.0
    running_label_loss = 0
    running_domain_loss = 0
    total_hit_sou, total_num_sou, total_hit_tar, total_num_tar = 0.0, 0.0, 0.0, 0.0
    test_acc_list = []
    train_start = time.time()

    for episode in range(5001):
        task_sou = utils.Task(metatrain_data, CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)
        support_dataloader_sou = utils.get_HBKC_data_loader(task_sou, num_per_class=SHOT_NUM_PER_CLASS, split="train",
                                                            shuffle=False)
        query_dataloader_sou = utils.get_HBKC_data_loader(task_sou, num_per_class=QUERY_NUM_PER_CLASS, split="test",
                                                          shuffle=False)

        supports_sou, support_labels_sou = support_dataloader_sou.__iter__().__next__()
        querys_sou, query_labels_sou = query_dataloader_sou.__iter__().__next__()

        query_real_labels_sou = task_sou.query_real_labels

        semantic_query_sou = torch.zeros(CLASS_NUM * QUERY_NUM_PER_CLASS, 768)
        semantic_query_sou_all = torch.zeros(CLASS_NUM * QUERY_NUM_PER_CLASS, 768)

        for i, class_id in enumerate(query_real_labels_sou):
            semantic_query_sou[i] = torch.from_numpy(semantic_mapping_sou[class_id])
            semantic_query_sou_all[i] = torch.from_numpy(semantic_mapping_sou_all)

        semantic_features_sou = semantic_feature_encoder(semantic_query_sou.cuda())
        semantic_features_sou_all = semantic_feature_encoder(semantic_query_sou_all.cuda())

        supports_sou = mapping_network(supports_sou.cuda(), domain='source')
        querys_sou = mapping_network(querys_sou.cuda(), domain='source')

        task_tar = utils.Task(target_da_metatrain_data, TEST_CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)
        support_dataloader_tar = utils.get_HBKC_data_loader(task_tar, num_per_class=SHOT_NUM_PER_CLASS, split="train",
                                                            shuffle=False)
        query_dataloader_tar = utils.get_HBKC_data_loader(task_tar, num_per_class=QUERY_NUM_PER_CLASS, split="test",
                                                          shuffle=False)

        supports_tar, support_labels_tar = support_dataloader_tar.__iter__().__next__()
        querys_tar, query_labels_tar = query_dataloader_tar.__iter__().__next__()

        query_real_labels_tar = task_tar.query_real_labels
        semantic_query_tar = torch.zeros(TEST_CLASS_NUM * QUERY_NUM_PER_CLASS, 768)
        semantic_query_tar_all = torch.zeros(TEST_CLASS_NUM * QUERY_NUM_PER_CLASS, 768)

        for i, class_id in enumerate(query_real_labels_tar):
            semantic_query_tar[i] = torch.from_numpy(semantic_mapping_tar[class_id])
            semantic_query_tar_all[i] = torch.from_numpy(semantic_mapping_tar_all)

        semantic_features_tar = semantic_feature_encoder(semantic_query_tar.cuda())
        semantic_features_tar_all = semantic_feature_encoder(semantic_query_tar_all.cuda())

        supports_tar = mapping_network(supports_tar.cuda(), domain='target')
        querys_tar = mapping_network(querys_tar.cuda(), domain='target')

        support_features_sou, supp_att_sou = feature_encoder(supports_sou.cuda())
        query_features_sou, query_att_sou = feature_encoder(querys_sou.cuda())
        query_features_tar, query_att_tar = feature_encoder(querys_tar.cuda())
        support_features_tar, supp_att_tar = feature_encoder(supports_tar.cuda())

        mix_querys_1_sou, query_label_11_sou, query_label_12_sou, lam_11_sou, lam_12_sou = cutmix1(querys_sou,
                                                                                                   query_labels_sou,
                                                                                                   query_att_sou,
                                                                                                   alpha=1.0)
        mix_querys_2_sou, query_label_21_sou, query_label_22_sou, lam_21_sou, lam_22_sou = cutmix2(querys_sou,
                                                                                                   query_labels_sou,
                                                                                                   query_att_sou,
                                                                                                   querys_tar,
                                                                                                   query_labels_tar,
                                                                                                   query_att_tar,
                                                                                                   alpha=1.0)

        mix_query_features_1_sou, mix_query_outputs_1_sou = feature_encoder(mix_querys_1_sou.cuda())
        mix_query_features_2_sou, mix_query_outputs_2_sou = feature_encoder(mix_querys_2_sou.cuda())

        mix_querys_1_tar, query_label_11_tar, query_label_12_tar, lam_11_tar, lam_12_tar = cutmix1(querys_tar,
                                                                                                   query_labels_tar,
                                                                                                   query_att_tar,
                                                                                                   alpha=1.0)
        mix_querys_2_tar, query_label_21_tar, query_label_22_tar, lam_21_tar, lam_22_tar = cutmix2(querys_tar,
                                                                                                   query_labels_tar,
                                                                                                   query_att_tar,
                                                                                                   querys_sou,
                                                                                                   query_labels_sou,
                                                                                                   query_att_sou,
                                                                                                   alpha=1.0)

        mix_query_features_1_tar, mix_query_outputs_1_tar = feature_encoder(mix_querys_1_tar.cuda())
        mix_query_features_2_tar, mix_query_outputs_2_tar = feature_encoder(mix_querys_2_tar.cuda())

        if SHOT_NUM_PER_CLASS > 1:
            support_proto_sou = support_features_sou.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)
            support_proto_tar = support_features_tar.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)
        else:
            support_proto_sou = support_features_sou
            support_proto_tar = support_features_tar

        logit_sou = euclidean_metric(query_features_sou, support_proto_sou)
        logits_1_sou = euclidean_metric(mix_query_features_1_sou, support_proto_sou)
        logits_2_sou = euclidean_metric(mix_query_features_2_sou, support_proto_sou)

        f_loss_sou = F.cross_entropy(logit_sou, query_labels_sou.long().cuda(), reduction="none")
        f_loss_1_sou = lam_11_sou * F.cross_entropy(logits_1_sou, query_label_11_sou.long().cuda(),
                                                    reduction="none") + lam_12_sou * F.cross_entropy(logits_1_sou,
                                                                                                     query_label_12_sou.long().cuda(),
                                                                                                     reduction="none")
        f_loss_2_sou = lam_21_sou * F.cross_entropy(logits_2_sou, query_label_21_sou.long().cuda(),
                                                    reduction="none") + lam_22_sou * F.cross_entropy(logits_2_sou,
                                                                                                     query_label_22_sou.long().cuda(),
                                                                                                     reduction="none")

        f_loss_sou = torch.mean(f_loss_sou, axis=0)
        f_loss_1_sou = torch.mean(f_loss_1_sou, axis=0)
        f_loss_2_sou = torch.mean(f_loss_1_sou, axis=0)

        logit_tar = euclidean_metric(query_features_tar, support_proto_tar)
        logits_1_tar = euclidean_metric(mix_query_features_1_tar, support_proto_tar)
        logits_2_tar = euclidean_metric(mix_query_features_2_tar, support_proto_tar)

        f_loss_tar = F.cross_entropy(logit_tar, query_labels_tar.long().cuda(), reduction="none")
        f_loss_1_tar = lam_11_tar * F.cross_entropy(logits_1_tar, query_label_11_tar.long().cuda(),
                                                    reduction="none") + lam_12_tar * F.cross_entropy(logits_1_tar,
                                                                                                     query_label_12_tar.long().cuda(),
                                                                                                     reduction="none")
        f_loss_2_tar = lam_21_tar * F.cross_entropy(logits_2_tar, query_label_21_tar.long().cuda(),
                                                    reduction="none") + lam_22_tar * F.cross_entropy(logits_2_tar,
                                                                                                     query_label_22_tar.long().cuda(),
                                                                                                     reduction="none")

        f_loss_tar = torch.mean(f_loss_tar, axis=0)
        f_loss_1_tar = torch.mean(f_loss_1_tar, axis=0)
        f_loss_2_tar = torch.mean(f_loss_1_tar, axis=0)

        lam_11_sou = lam_11_sou.unsqueeze(1)
        lam_12_sou = lam_12_sou.unsqueeze(1)
        semantic_features_sou_mix = lam_11_sou * semantic_features_sou + lam_12_sou * semantic_features_sou_all

        lam_11_tar = lam_11_tar.unsqueeze(1)
        lam_12_tar = lam_12_tar.unsqueeze(1)
        semantic_features_tar_mix = lam_11_tar * semantic_features_tar + lam_12_tar * semantic_features_tar_all

        text_align_loss_sou = image_text_contrastive_loss(mix_query_features_1_sou.cuda(),semantic_features_sou_mix.cuda())
        text_align_loss_tar = image_text_contrastive_loss(mix_query_features_1_tar.cuda(),semantic_features_tar_mix.cuda())

        joint_feature_sou = attn_layer( mix_query_features_1_sou.cuda(),semantic_features_sou_mix.cuda())
        joint_feature_tar = attn_layer(mix_query_features_1_tar.cuda(), semantic_features_tar_mix.cuda() )

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
        loss = f_loss + 2 * (f_loss_1 + f_loss_2) + 1 * (text_align_loss + f_loss_joint)

        feature_encoder.zero_grad()
        mapping_network.zero_grad()
        semantic_feature_encoder.zero_grad()
        loss.backward()
        feature_encoder_optim.step()
        mapping_network_optim.step()
        semantic_feature_encoder_optim.step()

        total_hit_sou += torch.sum(torch.argmax(logit_sou.cuda(), dim=1).cpu() == query_labels_sou).item()
        total_num_sou += querys_sou.shape[0]
        acc_sou = total_hit_sou / total_num_sou

        total_hit_tar += torch.sum(torch.argmax(logit_tar.cuda(), dim=1).cpu() == query_labels_tar).item()
        total_num_tar += querys_tar.shape[0]
        acc_tar = total_hit_tar / total_num_tar

        if (episode + 1) % 100 == 0:
            train_loss.append(loss.item())
            print('episode {:>3d}:  acc_sou: {:6.4f}, acc_tar: {:6.4f}, loss: {:6.4f}'.format(episode + 1, acc_sou, acc_tar, loss.item()))

        if (episode + 1) % 200 == 1 or episode == 0:
            with torch.no_grad():
                print("Testing ...")
                train_end = time.time()
                mapping_network.eval()
                feature_encoder.eval()
                semantic_feature_encoder.eval()

                total_rewards = 0
                counter = 0
                accuracies = []
                predict = np.array([], dtype=np.int64)
                labels = np.array([], dtype=np.int64)

                train_datas, train_labels = train_loader.__iter__().__next__()
                train_datas = mapping_network(Variable(train_datas).cuda(), domain='target')
                train_features, _ = feature_encoder(train_datas)

                max_value = train_features.max()
                min_value = train_features.min()
                train_features = (train_features - min_value) * 1.0 / (max_value - min_value)

                KNN_classifier = KNeighborsClassifier(n_neighbors=1, weights='distance')
                KNN_classifier.fit(train_features.cpu().detach().numpy(), train_labels)

                for test_datas, test_labels in test_loader:
                    batch_size = test_labels.shape[0]

                    test_datas = mapping_network(Variable(test_datas).cuda(), domain='target')
                    test_features, _ = feature_encoder(test_datas)
                    test_features = (test_features - min_value) * 1.0 / (max_value - min_value)
                    test_features = test_features.cpu()

                    predict_labels = KNN_classifier.predict(test_features.cpu().detach().numpy())
                    test_labels = test_labels.numpy()
                    all_features.append(test_features)
                    all_preds.append(predict_labels)
                    rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(batch_size)]
                    total_rewards += np.sum(rewards)
                    counter += batch_size

                    predict = np.append(predict, predict_labels)
                    labels = np.append(labels, test_labels)

                    accuracy = total_rewards / 1.0 / counter
                    accuracies.append(accuracy)

                test_accuracy = 100. * total_rewards / len(test_loader.dataset)
                print('\t\tepisode {} accuracy: {}/{} ({:.2f}%)\n'.format(episode + 1, total_rewards,
                                                                          len(test_loader.dataset),
                                                                          100. * total_rewards / len(
                                                                              test_loader.dataset)))
                test_end = time.time()
                mapping_network.train()
                feature_encoder.train()
                semantic_feature_encoder.train()

                if test_accuracy > last_accuracy:
                    torch.save(mapping_network.state_dict(),
                               str("./checkpoints/DFSL_mapping_network_" + "IP_" + str(iDataSet) + "iter_" + str(
                                   TEST_LSAMPLE_NUM_PER_CLASS) + "shot.pkl"))
                    torch.save(feature_encoder.state_dict(),
                               str("./checkpoints/DFSL_feature_encoder_" + "IP_" + str(iDataSet) + "iter_" + str(
                                   TEST_LSAMPLE_NUM_PER_CLASS) + "shot.pkl"))
                    print("save networks for episode:", episode + 1)
                    last_accuracy = test_accuracy
                    best_episdoe = episode

                    acc[iDataSet] = 100. * total_rewards / len(test_loader.dataset)
                    OA = acc
                    C = metrics.confusion_matrix(labels, predict)
                    A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float64)
                    k[iDataSet] = metrics.cohen_kappa_score(labels, predict)

                print('best episode:[{}], best accuracy={}'.format(best_episdoe + 1, last_accuracy))

        if test_accuracy > best_acc_all:
            best_predict_all = predict
            best_G, best_RandPerm, best_Row, best_Column, best_nTrain = G, RandPerm, Row, Column, nTrain

        print('iter:{} best episode:[{}], best accuracy={}'.format(iDataSet, best_episdoe + 1, last_accuracy))
        print('***********************************************************************************')

        AA = np.mean(A, 1)
        AAMean = np.mean(AA, 0)
        AAStd = np.std(AA)

        AMean = np.mean(A, 0)
        AStd = np.std(A, 0)

        OAMean = np.mean(acc)
        OAStd = np.std(acc)

        kMean = np.mean(k)
        kStd = np.std(k)

        print("-------------------------------")
        print("train time per DataSet(s): " + "{:.5f}".format(train_end - train_start))
        print("test time per DataSet(s): " + "{:.5f}".format(test_end - train_end))

        print("-------------------------------")
        print("average OA: " + "{:.2f}".format(OAMean) + " +- " + "{:.2f}".format(OAStd))
        print("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
        print("average kappa: " + "{:.4f}".format(100 * kMean) + " +- " + "{:.4f}".format(100 * kStd))

        print("-------------------------------")
        print("accuracy for each class: ")
        for i in range(CLASS_NUM):
            print("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))

        print("-------------------------------")
        best_iDataset = 0
        for i in range(len(acc)):
            print('{}:{}'.format(i, acc[i]))
            if acc[i] > acc[best_iDataset]:
                best_iDataset = i
        print('best acc all={}:{}'.format(i, acc[best_iDataset]))

# #################classification map################################
#
# for i in range(len(best_predict_all)):
#     best_G[best_Row[best_RandPerm[best_nTrain + i]]][best_Column[best_RandPerm[best_nTrain + i]]] = \
#     best_predict_all[i] + 1
#
# hsi_pic = np.zeros((best_G.shape[0], best_G.shape[1], 3))
# for i in range(best_G.shape[0]):
#     for j in range(best_G.shape[1]):
#         if best_G[i][j] == 0:
#             hsi_pic[i, j, :] = [0, 0, 0]
#         if best_G[i][j] == 1:
#             hsi_pic[i, j, :] = [0, 0, 1]
#         if best_G[i][j] == 2:
#             hsi_pic[i, j, :] = [0, 1, 0]
#         if best_G[i][j] == 3:
#             hsi_pic[i, j, :] = [0, 1, 1]
#         if best_G[i][j] == 4:
#             hsi_pic[i, j, :] = [1, 0, 0]
#         if best_G[i][j] == 5:
#             hsi_pic[i, j, :] = [1, 0, 1]
#         if best_G[i][j] == 6:
#             hsi_pic[i, j, :] = [1, 1, 0]
#         if best_G[i][j] == 7:
#             hsi_pic[i, j, :] = [0.5, 0.5, 1]
#         if best_G[i][j] == 8:
#             hsi_pic[i, j, :] = [0.65, 0.35, 1]
#         if best_G[i][j] == 9:
#             hsi_pic[i, j, :] = [0.75, 0.5, 0.75]
#         if best_G[i][j] == 10:
#             hsi_pic[i, j, :] = [0.75, 1, 0.5]
#         if best_G[i][j] == 11:
#             hsi_pic[i, j, :] = [0.5, 1, 0.65]
#         if best_G[i][j] == 12:
#             hsi_pic[i, j, :] = [0.65, 0.65, 0]
#         if best_G[i][j] == 13:
#             hsi_pic[i, j, :] = [0.75, 1, 0.65]
#         if best_G[i][j] == 14:
#             hsi_pic[i, j, :] = [0, 0, 0.5]
#         if best_G[i][j] == 15:
#             hsi_pic[i, j, :] = [0, 1, 0.75]
#         if best_G[i][j] == 16:
#             hsi_pic[i, j, :] = [0.5, 0.75, 1]
#
# utils.classification_map(hsi_pic[4:-4, 4:-4, :], best_G[4:-4, 4:-4], 24, "./F3IP_{}shot.png".format(TEST_LSAMPLE_NUM_PER_CLASS))

