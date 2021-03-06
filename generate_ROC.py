import argparse
import logging
import os
import torch
import torch.backends.cudnn as cudnn
import metric
from anomaly_detector_model import AnomalyDetector, RegularizedLoss, custom_objective
from features_loader import FeaturesLoaderVal
from lr_scheduler import MultiFactorScheduler
from model import model, static_model
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from os import path
import cv2
import numpy as np


parser = argparse.ArgumentParser(description="PyTorch Video Classification Parser")
parser.add_argument('--features_path', default='../dataset/i3d1/',
                    help="path to features")
parser.add_argument('--annotation_path', default="Test_Annotation.txt",
                    help="path to annotations")
parser.add_argument('--random-seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--model-dir', type=str, default="../anomaly_fusion/exps/model",
                    help="set logging file.")


if __name__ == "__main__":
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    data_loader = FeaturesLoaderVal(features_path=args.features_path,
                                    annotation_path=args.annotation_path)

    data_iter = torch.utils.data.DataLoader(data_loader,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=8,  # 4, # change this part accordingly
                                            pin_memory=True)

    network = AnomalyDetector()
    network.to(device)
    net = static_model(net=network,
                       criterion=RegularizedLoss(network, custom_objective).cuda(),
                       model_prefix=args.model_dir,
                       )
    model_path = net.get_checkpoint_path(100)
    net.load_checkpoint(pretrain_path=model_path, epoch=100)
    net.net.to(device)

    # net.net = torch.nn.DataParallel(net.net).cuda()

    # enable cudnn tune
    cudnn.benchmark = True

    y_trues = None
    y_preds = None
    v_pre=[]
    
    for features, start_end_couples, feature_subpaths, lengths in tqdm(data_iter):
        # features is a batch where each item is a tensor of 32 4096D features
        features = features.to(device)
        with torch.no_grad():
            input_var = torch.autograd.Variable(features)
            outputs,vids = net.predict(features)  # (batch_size, 32)
            outputs = outputs.reshape(outputs.shape[0], 32)
            for vid_len,vid, couples, output in zip(lengths, vids.cpu().numpy(),start_end_couples, outputs.cpu().numpy()):
                v_pre.append(vid)
                y_true = np.zeros(vid_len)
                segments_len = vid_len // 32
                for couple in couples:
                    if couple[0] != -1:
                        y_true[couple[0]: couple[1]] = 1
                y_pred = np.zeros(vid_len)
                for i in range(32):
                    segment_start_frame = i * segments_len
                    segment_end_frame = (i + 1) * segments_len
                    y_pred[segment_start_frame: segment_end_frame] = output[i]

                if y_trues is None:
                    y_trues = y_true
                    y_preds = y_pred
                else:
                    y_trues = np.concatenate([y_trues, y_true])
                    y_preds = np.concatenate([y_preds, y_pred])
    np.save('y_trues',y_trues)
    np.save('y_sh',y_preds)
    np.save('v_pre',v_pre)
    
    fpr, tpr, thresholds = roc_curve(y_true=y_trues, y_score=y_preds, pos_label=1)
    lw = 2
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")

    if not path.exists(r'graphs'):
        os.mkdir(r'graphs')
    plt.savefig(path.join('graphs', 'roc_sh.png'))
