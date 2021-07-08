import argparse
import os
import json
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.nn import functional as F
import torch.optim
import torch.utils.data

from dataset import TestDataset
from models import ModelBuilder, SegmentationModule
from lib.nn import user_scattered_collate


__all__ = ["default_argument_parser", "load_dict", "get_img_list",
           "load_seg_module", "load_test_data", "load_classes", "load_checkpoint",
           "img2onehot", "get_obj_onehot_vector", "classify_step"]

logging.basicConfig(level=logging.DEBUG)


def default_argument_parser():
    parser = argparse.ArgumentParser(
        description="PyTorch OTS Testing"
    )
    parser.add_argument(
        "--imgs",
        default='/home/xxx/Documents/data/places7/val/',
        type=str,
        help="dataset"
    )
    parser.add_argument(
        "--cfg",
        default="config/resnet.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--cls-file",
        default="data/places365_7.txt",
        metavar="FILE",
        help="path to class file",
        type=str,
    )
    parser.add_argument(
        "--ckpt",
        default="ckpt/resnet50_7.pth.tar",
        metavar="FILE",
        help="path to checkpoint",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    parser.add_argument(
        "--local-object",
        default="",
        help="path to local object features",
        type=str,
    )
    
    return parser


def load_dict(filename):
    try:
        with open(filename, 'r') as json_file:
            dic = json.load(json_file)
        return dic
    except:
        logging.info('json file error {}'.format(filename))
        return {} 


def get_img_list(path):
    is_image_file = lambda x : any(x.endswith(extension)
                                   for extension in ['.jpg', 'png', 'gif', 'bmp'])
    return [os.path.join(r, i) for r, _, f in os.walk(path) for i in f if is_image_file(i)]


def load_seg_module(cfg):
    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder,
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder,
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)
    segmentation_module = SegmentationModule(net_encoder, net_decoder, nn.NLLLoss(ignore_index=-1))
    return segmentation_module.cuda().eval()


def load_test_data(cfg):
    dataset_test = TestDataset(
        cfg.list_test,
        cfg.DATASET)
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=cfg.TEST.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=12,
        drop_last=True)
    return loader_test


def load_classes(class_file):
    with open(class_file) as f:
        classes = []
        for line in f:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            gate, class_name = int(line[0]), line[1]
            if gate > 0:
                classes.append(class_name)
    classes = tuple(classes)
    num_classes = len(classes)
    logging.info('class number is {}'.format(num_classes))
    return classes, num_classes


def load_checkpoint(checkpoint, obj_model, classifier):
    checkpoint = torch.load(checkpoint)
    if classifier:
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
    if obj_model:
        obj_model.load_state_dict(checkpoint['obj_state_dict'])
    return obj_model, classifier


def img2onehot(img_name, one_hot):
    img_name = img_name.split('/')
    p = os.path.join(one_hot, img_name[-3], img_name[-2], img_name[-1].split('.')[0] + '.json')
    return p


def get_obj_onehot_vector(img_name, one_hot):
    one_hot_path = img2onehot(img_name, one_hot)
    v = load_dict(one_hot_path)
    img_name = list(v.keys())[0]
    return v[img_name]


def classify_step(logit, classes):
    class_vector = F.softmax(logit, 1).data.squeeze()
    assert len(class_vector) == len(classes), "class number must match"
    probs, idx = class_vector.sort(0, True)
    result = classes[idx[0]]
    return result
