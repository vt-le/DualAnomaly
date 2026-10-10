import os
import time
import json
import joblib
import pickle
import argparse
import datetime

import numpy
import numpy as np
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn

# config
from cfgs.default import get_config
# dataset
from data.chunked_dataset import chunked_sample_loader

# model
from models.dual_anomaly_model import AnomalyDetection

# utils
from utils.logger import create_test_logger
from utils.eval_utils import save_evaluation_curves


METADATA = {
    "ped1": {
        "testing_video_num": 36,
        "testing_frames_cnt": [200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
                               200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
                               200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
                               200, 200, 200, 200, 200, 200
                               ]
    },
    "ped2": {
        "testing_video_num": 12,
        "testing_frames_cnt": [180, 180, 150, 180, 150, 180, 180, 180, 120, 150,
                               180, 180]
    },
    "avenue": {
        "testing_video_num": 21,
        "testing_frames_cnt": [1439, 1211, 923, 947, 1007, 1283, 605, 36, 1175, 841,
                               472, 1271, 549, 507, 1001, 740, 426, 294, 248, 273,
                               76],
    },
    "shanghaitech": {
        "testing_video_num": 107,
        "testing_frames_cnt": [265, 433, 337, 601, 505, 409, 457, 313, 409, 337,
                               337, 457, 577, 313, 529, 193, 289, 289, 265, 241,
                               337, 289, 265, 217, 433, 409, 529, 313, 217, 241,
                               313, 193, 265, 317, 457, 337, 361, 529, 409, 313,
                               385, 457, 481, 457, 433, 385, 241, 553, 937, 865,
                               505, 313, 361, 361, 529, 337, 433, 481, 649, 649,
                               409, 337, 769, 433, 241, 217, 265, 265, 217, 265,
                               409, 385, 481, 457, 313, 601, 241, 481, 313, 337,
                               457, 217, 241, 289, 337, 313, 337, 265, 265, 337,
                               361, 433, 241, 433, 601, 505, 337, 601, 265, 313,
                               241, 289, 361, 385, 217, 337, 265]
    },

}
# Ped2:         --cfg cfgs/ped2/ped2_DualAnomaly.yaml
# Avenue:       --cfg cfgs/avenue/avenue_DualAnomaly.yaml
# ShanghaiTech: --cfg cfgs/shanghaitech/shanghaitech_DualAnomaly.yaml

def parse_option():
    parser = argparse.ArgumentParser('Project 3', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--gpus', type=int, help='GPU index')

    # distributed training
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def evaluate(config, logger, model_path, training_stats_path, suffix="best"):
    # Create a folder that will be stored evaluation results
    eval_dir = os.path.join(config.OUTPUT, config.EVAL)
    os.makedirs(eval_dir, exist_ok=True)

    testset_num_frames = np.sum(METADATA[config.DATA.DATASET]["testing_frames_cnt"])

    device = config.GPUS[0]

    model = AnomalyDetection(config=config, logger=logger, is_trained=False).to(device).eval()

    if training_stats_path is not None:
        training_scores_stats = torch.load(training_stats_path, weights_only=False)

        of_mean, of_std = np.mean(training_scores_stats["of_training_stats"]), \
                          np.std(training_scores_stats["of_training_stats"])
        frame_mean, frame_std = np.mean(training_scores_stats["frame_training_stats"]), \
                                np.std(training_scores_stats["frame_training_stats"])

    if isinstance(model_path, str):
        checkpoint = torch.load(model_path, map_location='cpu')
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        logger.warning(msg)
        logger.info(f"=> loaded successfully '{model_path}'")

        del checkpoint
        torch.cuda.empty_cache()

    dataloader = chunked_sample_loader(config, is_train=False)

    score_func = torch.nn.MSELoss(reduction='none')

    # bbox anomaly scores for each frame
    frame_bbox_scores = [{} for i in range(testset_num_frames.item())]

    start = time.time()
    for idx, test_data in tqdm(enumerate(dataloader), desc='Eval: ', total=len(dataloader)):
        sample_frames_test, sample_ofs_test, bbox_test, pred_frame_test, indices_test = test_data
        sample_tar = (sample_frames_test[-1]).to(device)
        sample_fra = (sample_frames_test[:-1])
        sample_fra = [sample.to(device) for sample in sample_fra]

        sample_of = sample_ofs_test.to(device)

        frame_out_test, flow_out_test = model(sample_fra, sample_of)

        # TODO: choose one of two approaches
        # Frame
        loss_frame = score_func((frame_out_test+1)/2, (sample_tar+1)/2)
        frame_scores = []
        for i in range(loss_frame.shape[0]):
            score = torch.mean(loss_frame[i])
            frame_scores.append(score.cpu().data.numpy())
        frame_scores = numpy.array(frame_scores)
        # Optical flow
        loss_flow = score_func((flow_out_test + 1) / 2, (sample_of + 1) / 2)
        flow_scores = []
        for i in range(loss_flow.shape[0]):
            score = torch.mean(loss_flow[i])
            flow_scores.append(score.cpu().data.numpy())
        flow_scores = numpy.array(flow_scores)

        '''
        loss_frame = score_func(out_test, sample_tar).cpu().data.numpy()
        frame_scores = np.sum(np.sum(np.sum(loss_frame, axis=3), axis=2), axis=1)
        '''

        if training_stats_path is not None:
            # mean-std normalization
            flow_scores = (flow_scores - of_mean) / of_std
            frame_scores = (frame_scores - frame_mean) / frame_std

        scores = config.TEST.FLOW_WEIGHT * flow_scores + config.TEST.FRAME_WEIGHT * frame_scores

        for i in range(len(scores)):
            frame_bbox_scores[pred_frame_test[i][-1].item()][i] = scores[i]

    # frame-level anomaly score
    frame_scores = np.empty(len(frame_bbox_scores))
    for i in range(len(frame_scores)):
        if len(frame_bbox_scores[i].items()) == 0:
            frame_scores[i] = 0
        else:
            frame_scores[i] = np.max(list(frame_bbox_scores[i].values()))

    joblib.dump(frame_scores, os.path.join(config.OUTPUT, config.EVAL, "frame_scores_%s.json" % suffix))

    # ================== Calculate AUC ==============================
    # load gt labels
    gt_label = open(os.path.join(config.DATA.DATA_PATH, config.DATA.DATASET, "ground_truth_demo/gt_label.json"), "rb")
    gt = pickle.load(gt_label, encoding='bytes')
    # gt = np.load(os.path.join(config.DATA.DATA_PATH, config.DATA.DATASET, "ground_truth_demo/gt_label_new.npy"), allow_pickle=True)
    gt_concat = np.concatenate(list(gt.values()), axis=0)
    # gt_concat = np.concatenate(list(gt.item()), axis=0)

    new_gt = np.array([])
    new_frame_scores = np.array([])

    start_idx = 0
    for cur_video_id in range(METADATA[config.DATA.DATASET]["testing_video_num"]):
        gt_each_video = gt_concat[start_idx:start_idx + METADATA[config.DATA.DATASET]["testing_frames_cnt"][cur_video_id]][4:]
        scores_each_video = frame_scores[
                            start_idx:start_idx + METADATA[config.DATA.DATASET]["testing_frames_cnt"][cur_video_id]][4:]

        start_idx += METADATA[config.DATA.DATASET]["testing_frames_cnt"][cur_video_id]

        new_gt = np.concatenate((new_gt, gt_each_video), axis=0)
        new_frame_scores = np.concatenate((new_frame_scores, scores_each_video), axis=0)

    gt_concat = new_gt
    frame_scores = new_frame_scores

    curves_save_path = os.path.join(config.OUTPUT, config.EVAL, f"anomaly_curves_{suffix}")
    auc = save_evaluation_curves(frame_scores, gt_concat, curves_save_path,
                                 np.array(METADATA[config.DATA.DATASET]["testing_frames_cnt"]) - 4)

    test_time = time.time() - start
    logger.info(f'Dataset {config.DATA.DATASET} takes {datetime.timedelta(seconds=int(test_time))}')

    return auc


if __name__ == '__main__':
    args, config = parse_option()

    cudnn.benchmark = True         # "True" in Swin
    cudnn.deterministic = False      # Not used in Swin
    cudnn.enabled = True           # TRUE or FALSE?

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_test_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")

    # print config
    # logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    training_stats_path = os.path.join(config.OUTPUT, f'training_stats.npy')

    # model_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{0}.pth')
    model_path = os.path.join(config.OUTPUT, f'best.pth')
    # testing_folder = os.path.join(config.DATA.DATA_PATH, config.DATA.DATASET, "testing", config.DATA.CHUNKED_FOLDER)
    auc = evaluate(config, logger, model_path=model_path, training_stats_path=training_stats_path, suffix="best") * 100

    logger.info(f"AUC: {auc:.1f}%")

