import os
import time
import json
import random
import inspect
import argparse
import datetime
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn

# config
from cfgs.default import get_config
# dataset
from data.chunked_dataset import chunked_sample_loader
# model
from models.dual_anomaly_model import AnomalyDetection

# utils
from utils.logger import create_logger
from utils.optimizer import build_optimizer
from utils.scheduler import build_scheduler
from utils.losses import Losses
from utils.anomaly_score import psnr_park
from utils.checkpoint import save_checkpoint, save_best_checkpoint

# visualization
from visualization.visualize import Visualize, tensor2im

# evaluate
from hybrid_eval import evaluate


def parse_option():
    parser = argparse.ArgumentParser('Dual Anomaly', add_help=False)
    parser.add_argument('--cfg', type=str, default="cfgs/avenue/avenue_DualAnomaly.yaml",
                        metavar="FILE", help='path to config file')
    # --cfg cfgs/ped2/ped2_DualAnomaly.yaml

    # --cfg cfgs/avenue/avenue_DualAnomaly.yaml

    # --cfg cfgs/shanghaitech/shanghaitech_DualAnomaly.yaml


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


def main(config, logger):
    dataloader = chunked_sample_loader(config=config, is_train=True)
    logger.info(f"The number of input sequences of {config.DATA.DATASET.upper()} dataset: {len(dataloader)}")

    model = AnomalyDetection(config=config, logger=logger, is_trained=True)

    logger.info(f'Model: {inspect.getfile(AnomalyDetection)}')

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters:,}")

    gpus = list(config.GPUS)
    model = torch.nn.DataParallel(model, device_ids=gpus)  # .cuda()
    model.cuda()
    model_without_ddp = model.module

    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(dataloader))

    criterion = Losses(config).cuda()
    flow_criterion = torch.nn.MSELoss().cuda()
    mse = torch.nn.MSELoss(reduction='none')
    visualizer = Visualize(config=config, final_output_dir=config.OUTPUT)

    logger.info("Start training")
    start_time = time.time()
    best_auc = -1
    count = 0
    topk_auc = []
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        train_one_epoch(config, model, criterion, flow_criterion, dataloader, optimizer, epoch, lr_scheduler, mse, logger, visualizer)

        if epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1):
            save_checkpoint(config, epoch, model_without_ddp, optimizer, lr_scheduler, logger, resume=False)

            # computing training stats
            stats_save_path = os.path.join(config.OUTPUT, f'training_stats.npy{epoch + 1}')
            training_stats(config, os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth'), stats_save_path)

            with torch.no_grad():
                model_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
                auc = evaluate(config, logger, model_path, stats_save_path, suffix=str(epoch))

                logger.info(f"==================================================================")
                logger.info(f"AUC (E: {epoch}): {auc} \t Best AUC: {best_auc}")
                logger.info(f"==================================================================")

                if len(topk_auc) > 4:
                    os.remove(os.path.join(config.OUTPUT, topk_auc[0]['ckpt_name']))
                    topk_auc.pop(0)
                topk_auc.append({'ckpt_name': f'ckpt_epoch_{epoch}.pth', 'auc': auc})
                topk_auc = sorted(topk_auc, key=lambda d: d['auc'])

                if auc > best_auc:
                    best_auc = auc
                    count = 0
                    save_best_checkpoint(config, model_without_ddp)
                    # only_model_saver(model.state_dict(), os.path.join(paths["ckpt_dir"], "best.pth"))
                else:
                    count = count + 1
                if count == config.TRAIN.EARLY_STOPPING:
                    break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def training_stats(config, ckpt_path, stats_save_path):
    device = config.GPUS[0]

    model = AnomalyDetection(config=config, logger=logger, is_trained=False).to(device).eval()

    if isinstance(ckpt_path, str):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        logger.warning(msg)
        logger.info(f"=> loaded successfully '{ckpt_path}'")

        del checkpoint
        torch.cuda.empty_cache()

    score_func = torch.nn.MSELoss(reduction="none")
    dataloader = chunked_sample_loader(config=config, is_train=True)

    of_training_stats = []
    frame_training_stats = []

    with torch.no_grad():
        for samples in tqdm(dataloader, desc="Stats: ", total=len(dataloader)):
            sample_frames, sample_flows, _, _, _ = samples

            sample_tar = (sample_frames[-1]).cuda(non_blocking=True)
            sample_frm = sample_frames[:-1]
            sample_frm = [sample.cuda(non_blocking=True) for sample in sample_frm]

            sample_flows = sample_flows.cuda(non_blocking=True)

            frame_outputs, flow_output = model(sample_frm, sample_flows)

            loss_flow = score_func((flow_output+1)/2, (sample_flows+1)/2)
            flow_scores = []
            for i in range(loss_flow.shape[0]):
                score = torch.mean(loss_flow[i])
                flow_scores.append(score.cpu().data.numpy())
            flow_scores = np.array(flow_scores)

            loss_frame = score_func((frame_outputs + 1) / 2, (sample_tar + 1) / 2)
            frame_scores = []
            for i in range(loss_frame.shape[0]):
                score = torch.mean(loss_frame[i])
                frame_scores.append(score.cpu().data.numpy())
            frame_scores = np.array(frame_scores)

            of_training_stats.append(flow_scores)
            frame_training_stats.append(frame_scores)

    print("=========Forward pass for training stats done!==========")
    of_training_stats = np.concatenate(of_training_stats, axis=0)
    frame_training_stats = np.concatenate(frame_training_stats, axis=0)

    training_stats = dict(of_training_stats=of_training_stats,
                          frame_training_stats=frame_training_stats)
    # save to file
    torch.save(training_stats, stats_save_path)


def train_one_epoch(config, model, criterion, flow_criterion, data_loader, optimizer, epoch, lr_scheduler, mse, logger, visualizer):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)

    start = time.time()
    for idx, samples in enumerate(data_loader):
        sample_frames, sample_flows, _, _, _ = samples

        sample_tar = (sample_frames[-1]).cuda(non_blocking=True)
        sample_frm = sample_frames[:-1]
        sample_frm = [sample.cuda(non_blocking=True) for sample in sample_frm]

        sample_flows = sample_flows.cuda(non_blocking=True)

        frame_outputs, flow_output = model(sample_frm, sample_flows)

        flow_l = flow_criterion(flow_output, sample_flows)
        int_l, gra_l, mss_l, l2_l = criterion(frame_outputs, sample_tar)
        loss = int_l + gra_l + mss_l + l2_l + flow_l

        # compute PSNR
        mse_imgs = torch.mean(mse((frame_outputs + 1) / 2, (sample_tar + 1) / 2)).item()
        psnr = psnr_park(mse_imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'mem {memory_used:.0f}MB\t'
                f'lr {lr:.6f}\t'
                f'psnr {psnr:.2f}\t'
                f'int_l {int_l:.6f}\t'
                f'gra_l {gra_l:.6f}\t'
                f'mss_l {mss_l:.6f}\t'
                f'l2_l {l2_l:.6f}\t'
                f'flow_l {flow_l:.6f}\t'
            )

            # display output image
            visual_list = [('input1', tensor2im(sample_frm[0][0])),
                           ('input2', tensor2im(sample_frm[1][0])),
                           ('input3', tensor2im(sample_frm[2][0])),
                           ('input4', tensor2im(sample_frm[3][0])),
                           ('target', tensor2im(sample_tar[0])),
                           ('OUTPUT', tensor2im(frame_outputs[0]))]
            visuals = OrderedDict(visual_list)
            visualizer.display_current_results(visuals, epoch, idx)

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


if __name__ == '__main__':
    args, config = parse_option()

    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True
    # os.environ["PYTHONHASHSEED"] = str(seed)

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(config, logger)
