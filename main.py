# System libs
import os
import time
from tqdm import tqdm
import logging

# Our libs
from lib.utils import as_numpy
from config import cfg
from functions import *
from models import *

os.environ["CUDA_VISIBLE_DEVICES"]="0"
logging.basicConfig(level=logging.DEBUG)
logging.info('current time is {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

class main(object):
    def __init__(self, cfg, args):
        self.classes, self.classes_num = load_classes(args.cls_file)
        self.seg_model = load_seg_module(cfg)
        self.test_data = load_test_data(cfg)
        self.obj_model = OAM_GRAM(in_dim=1024, one_hot_cls_num=150).cuda().eval()
        self.classifier = Classifier(num_classes=self.classes_num, in_dim=2048).cuda().eval()
        if args.ckpt:
            self.obj_model, self.classifier = \
                load_checkpoint(args.ckpt, self.obj_model, self.classifier)
        self.cfg = cfg
        self.args = args
        self.correct = 0
        self.count = 0
        logging.info(self.obj_model)
        logging.info(self.classifier)

    def get_object_feature(self, segSize, img_resized_list, batch_data):
        segmentation_module = self.seg_model
        args = self.args
        cfg = self.cfg

        # Upload object features from local file instead of calculating the online
        if args.local_object:
            object_feature = get_obj_onehot_vector(batch_data['info'], args.local_object)
            object_feature = torch.FloatTensor([object_feature]).cuda().view(1,1024,150,1)
            return object_feature

        with torch.no_grad():
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1]).cuda()
            feature = torch.zeros(1, 1024, segSize[0], segSize[1]).cuda()
            channels = feature.shape[1]

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img.cuda()
                del feed_dict['img_ori']
                del feed_dict['info']
                pred_tmp, pred_tmp_feature_map = segmentation_module(feed_dict, segSize=segSize)
                feature = feature + pred_tmp_feature_map / len(cfg.DATASET.imgSizes)
                scores = scores + pred_tmp / len(cfg.DATASET.imgSizes)
            
            # Uncomment the following codes to verify the correctness of following codes.
            # scores = torch.Tensor([[[[1, 0], [1, 0]], [[0, 1], [0, 0]], [[0, 0], [0, 1]]]])
            # feature = torch.Tensor([[[[0.5, 1], [0.5, 2]], [[0.5, 1], [0.5, 2]], [[0.5, 1], [0.5, 2]], [[0.5, 1], [0.5, 2]], [[0.5, 1], [0.5, 2]]]])
            # channels = 5
            # cfg.DATASET.num_class = 3


            # ***Open trigger will double the whole inference speed.****
            # ***However, this trigger will also slightly influence the object feature value** 
            # This inconsistency issue is stemmed from ***Pytorch***, not our method.
            # Even the object feature value will slightly changed, our model is still stable.
            trigger = True
            # Object Feature Aggregation
            if trigger:
                scores = scores.view(cfg.DATASET.num_class, -1)
                s, pred = torch.max(scores, dim=0)
                object_feature = torch.zeros(cfg.DATASET.num_class, channels)
                feature = feature.view(channels, -1).permute((1, 0))
                for i in range(cfg.DATASET.num_class):
                    idx = torch.where(pred == i)
                    score = s[idx]
                    total_score = score.sum()
                    if total_score <= 0:
                        continue
                    chosen = feature[idx]
                    vec = torch.sum(chosen * score.view(-1, 1), dim=0) / total_score
                    object_feature[i] = vec
                object_feature = as_numpy(object_feature)
            else:
                s, pred = torch.max(scores, dim=1)
                object_feature = torch.zeros(cfg.DATASET.num_class, channels).cuda()
                pred_vec = pred.view(1, -1)
                pred_mat = pred_vec.repeat(channels, 1)
                s = s.view(1, -1)
                feature = feature.view(channels, -1)
                for i in range(cfg.DATASET.num_class):
                    m_vec = torch.eq(pred_vec, i)
                    m_mat = torch.eq(pred_mat, i)
                    score = s[m_vec].view(-1, 1)
                    if score.sum() <= 0:
                        continue
                    chosen = feature[m_mat].view(channels, -1)
                    vec = torch.mm(chosen, score) / score.sum()
                    object_feature[i] = vec.view(1, -1)
                object_feature = as_numpy(object_feature)

        object_feature = torch.FloatTensor([object_feature]).cuda().view(1,1024,150,1)
        return object_feature

    def test(self):
        pbar = tqdm(total=len(self.test_data))
        for batch_data in self.test_data:
            batch_data = batch_data[0]
            # segSize = (batch_data['img_ori'].shape[0], batch_data['img_ori'].shape[1])
            segSize = (256, 256)
            img_resized_list = batch_data['img_data']
            true_label = batch_data['info'].split('/')[-2]
            object_feature = self.get_object_feature(segSize, img_resized_list, batch_data)
            object_feature = self.obj_model(object_feature)
            logit = self.classifier(object_feature)
            pred_label = classify_step(logit, self.classes)
            
            if pred_label == true_label:
                self.correct += 1
            self.count += 1

            pbar.update(1)

        acc = 100 * self.correct / float(self.count)
        logging.info('Accuracy is {:2.2f}%, sample number is {}'.format(acc, self.count))

        return "Finished!"


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    logging.info(args)
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    assert os.path.exists(cfg.MODEL.weights_encoder) and \
        os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"
    if os.path.isdir(args.imgs):
        imgs = get_img_list(args.imgs)
    else:
        imgs = [args.imgs]
    assert len(imgs), "imgs should be a path to image (.jpg) or directory."
    cfg.list_test = [{'fpath_img': x} for x in imgs]
    #cfg.freeze()

    # test part
    model = main(cfg, args)
    model.test()
