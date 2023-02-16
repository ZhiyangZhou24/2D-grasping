import argparse
import datetime
import json
import logging
import os
import sys

import cv2
import numpy as np
import tensorboardX
import torch
import torch.optim as optim
import torch.utils.data
from torchsummary import summary

from hardware.device import get_device
from inference.models import get_network
from inference.post_process import post_process_output
from utils.data.cornell_data import CornellDataset
from utils.data import get_dataset
from utils.dataset_processing import evaluation
from utils.visualisation.gridshow import gridshow
from ranger import Ranger  # this is from ranger.py
from test_lr_curve import flat_and_anneal_lr_scheduler


def parse_args():
    parser = argparse.ArgumentParser(description='Train network')

    # Network
    parser.add_argument('--network', type=str, default='grconvnet3_imp_dwc1',
                        help='Network name in inference/models  grconvnet')
    parser.add_argument('--input-size', type=int, default=320,
                        help='Input image size for the network')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb', type=int, default=0,
                        help='Use RGB image for training (1/0)')
    parser.add_argument('--use-dropout', type=int, default=1,
                        help='Use dropout for training (1/0)')
    parser.add_argument('--dropout-prob', type=float, default=0.2,
                        help='Dropout prob for training (0-1)')
    parser.add_argument('--channel-size', type=int, default=32,
                        help='Internal channel size for the network')
    parser.add_argument('--iou-threshold', type=float, default=0.25,
                        help='Threshold for IOU matching')
    parser.add_argument('--iou-abla', type=bool, default=False,
                        help='Threshold albation for evaluation, need more time')
    
    parser.add_argument('--use-mish', type=bool, default=True,
                        help='(  True  False  )')
    parser.add_argument('--posloss', type=bool, default=True,
                        help='(  True  False  )')
    parser.add_argument('--upsamp', type=str, default='use_bilinear',
                        help='Use upsamp type (  use_duc  use_convt use_bilinear  )')
    parser.add_argument('--att', type=str, default='use_coora',
                        help='Use att type (  use_eca  use_se use_coora use_cba)')
    parser.add_argument('--use_gauss_kernel', type=float, default= 0.0,
                        help='Dataset gaussian progress 0.0 means not use gauss')
    parser.add_argument('--data-aug', type=bool, default=True,
                        help='Threshold albation for evaluation, need more time')

    # Datasets
    # /media/lab/ChainGOAT/Jacquard
    # /media/lab/e/zzy/datasets/Cornell
    parser.add_argument('--dataset', type=str,default='jacquard',
                        help='Dataset Name ("cornell" or "jacquard")')
    parser.add_argument('--dataset-path', type=str,default='/media/lab/ChainGOAT/Jacquard',
                        help='Path to dataset')
    parser.add_argument('--alfa', type=int, default=1,
                        help='len(Dataset)*alfa')
    parser.add_argument('--split', type=float, default=0.90,
                        help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-shuffle', action='store_true', default=False,
                        help='Shuffle the dataset')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split')
    parser.add_argument('--num-workers', type=int, default=16,
                        help='Dataset workers')

    # Training
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=0, help='权重衰减 L2正则化系数')
    parser.add_argument('--epochs', type=int, default=60,
                        help='Training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=1600,
                        help='Batches per Epoch')
    parser.add_argument('--optim', type=str, default='ranger',
                        help='Optmizer for the training. (adam or SGD)')
    parser.add_argument('--scheduler', type=str, default='flat',
                        help='scheduler for the training. (flat or multi-step or fixe)')
    parser.add_argument('--vis-lr', type=bool, default=True,
                        help='scheduler for the seeing. (False or True)')
    # Logging etc.
    parser.add_argument('--description', type=str, default='dwc1_d_bili_mish_coora32_drop2_ranger_bina_pos1',
                        help='Training description')
    parser.add_argument('--logdir', type=str, default='logs/jacquard_dwc',
                        help='Log directory')
    parser.add_argument('--vis', action='store_true',
                        help='Visualise the training process')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')
    parser.add_argument('--random-seed', type=int, default=123,
                        help='Random seed for numpy')
    parser.add_argument('--goon-train', type=bool, default=False, help='是否从已有网络继续训练')
    parser.add_argument('--model', type=str, default='logs/jacquard_dwc/230215_0137_dwc1_d_bili_mish_coora32_drop2_ranger_bina_pos1/epoch_05_iou_0.9358', help='保存的模型')
    parser.add_argument('--start-epoch', type=int, default=4, help='继续训练开始的epoch')
    args = parser.parse_args()
    return args


steps = []
lrs = []
epoch_lrs = []
global_step = 0

def validate(net, device, val_data, iou_threshold):
    """
    Run validation.
    :param net: Network
    :param device: Torch device
    :param val_data: Validation Dataset
    :param iou_threshold: IoU threshold
    :return: Successes, Failures and Losses
    """
    net.eval()

    results = {
        'correct': 0,
        'failed': 0,
        'loss': 0,
        'losses': {

        }
    }

    ld = len(val_data)

    with torch.no_grad():
        for x, y, didx, rot, zoom_factor in val_data:
            xc = x.to(device)
            yc = [yy.to(device) for yy in y]
            lossd = net.compute_loss(xc, yc,pos_loss=True)

            loss = lossd['loss']

            results['loss'] += loss.item() / ld
            for ln, l in lossd['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item() / ld

            q_out, ang_out, w_out = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                        lossd['pred']['sin'], lossd['pred']['width'])

            s = evaluation.calculate_iou_match(q_out,
                                               ang_out,
                                               val_data.dataset.get_gtbb(didx, rot, zoom_factor),
                                               no_grasps=1,
                                               grasp_width=w_out,
                                               threshold=iou_threshold
                                               )

            if s:
                results['correct'] += 1
            else:
                results['failed'] += 1

    return results


def train(epoch, net, device, train_data, scheduler, batches_per_epoch, vis=False):
    """
    Run one training epoch
    :param epoch: Current epoch
    :param net: Network
    :param device: Torch device
    :param train_data: Training Dataset
    :param scheduler: scheduler
    :param batches_per_epoch:  Data batches to train on
    :param vis:  Visualise training progress
    :return:  Average Losses for Epoch
    """
    results = {
        'loss': 0,
        'losses': {
        }
    }

    net.train()

    batch_idx = 0
    global steps
    global lrs
    global epoch_lrs
    global global_step
    # Use batches per epoch to make training on different sized datasets (cornell/jacquard) more equivalent.
    while batch_idx <= batches_per_epoch:
        for batch in range(batches_per_epoch):
            batch_idx += 1
            if batch_idx >= batches_per_epoch:
                break

            # lossd = net.compute_loss(xc, yc,pos_loss=True)

            # loss = lossd['loss']

            # if batch_idx % 100 == 0:

            # results['loss'] += loss.item()
            # for ln, l in lossd['losses'].items():
            #     if ln not in results['losses']:
            #         results['losses'][ln] = 0
            #     results['losses'][ln] += l.item()

            # scheduler.zero_grad()
            # loss.backward()
            # if global_step == 0 or (len(lrs) >= 1 and cur_lr != lrs[-1]):
            
            steps.append(global_step)
            cur_lr = scheduler.get_lr()[0]
            lrs.append(cur_lr)
            global_step+=1
            print("epoch {}, batch: {}, global_step:{} lr: {}".format(epoch, batch, global_step, cur_lr))
            scheduler.step()

            # Display the images
            # if vis:
            #     imgs = []
            #     n_img = min(4, x.shape[0])
            #     for idx in range(n_img):
            #         imgs.extend([x[idx,].numpy().squeeze()] + [yi[idx,].numpy().squeeze() for yi in y] + [
            #             x[idx,].numpy().squeeze()] + [pc[idx,].detach().cpu().numpy().squeeze() for pc in
            #                                           lossd['pred'].values()])
            #     gridshow('Display', imgs,
            #              [(xc.min().item(), xc.max().item()), (0.0, 1.0), (0.0, 1.0), (-1.0, 1.0),
            #               (0.0, 1.0)] * 2 * n_img,
            #              [cv2.COLORMAP_BONE] * 10 * n_img, 10)
            #     cv2.waitKey(2)

    results['loss'] /= batch_idx
    for l in results['losses']:
        results['losses'][l] /= batch_idx

    return results


def run():
    global steps
    global lrs
    global epoch_lrs
    global global_step
    args = parse_args()

    # Set-up output directories
    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    net_desc = '{}_{}'.format(dt, '_'.join(args.description.split()))

    save_folder = os.path.join(args.logdir, net_desc)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    tb = tensorboardX.SummaryWriter(save_folder)

    # Save commandline args
    if args is not None:
        params_path = os.path.join(save_folder, 'commandline_args.json')
        with open(params_path, 'w') as f:
            json.dump(vars(args), f)

    # Initialize logging
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        filename="{0}/{1}.log".format(save_folder, 'log'),
        format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    # Get the compute device
    device = get_device(args.force_cpu)

    # Load Dataset
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)
    logging.info('Dataset augmentation is {}'.format(args.data_aug))
    dataset = Dataset(args.dataset_path,
                      output_size=args.input_size,
                      ds_rotate=args.ds_rotate,
                      alfa=args.alfa,
                      random_rotate=args.data_aug,
                      random_zoom=args.data_aug,
                      include_depth=args.use_depth,
                      include_rgb=args.use_rgb,
                      use_gauss_kernel = args.use_gauss_kernel)
    logging.info('Dataset size is {}'.format(dataset.length))

    # Creating data indices for training and validation splits
    if args.dataset.title() != 'Jacquard':
        indices = list(range(dataset.len))
        logging.info('alfaed Dataset len is {}'.format(dataset.len))
        split = int(np.floor(args.split * dataset.len))
        if args.ds_shuffle: # 对应 imgwise 否则为obj
            np.random.seed(args.random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[:split], indices[split:]
    else:
        indices = list(range(dataset.length))
        logging.info('Dataset len is {}'.format(dataset.length))
        split = int(np.floor(args.split * dataset.length))
        if args.ds_shuffle: # 对应 imgwise 否则为obj
            np.random.seed(args.random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[:split], indices[split:]
    
    logging.info('Training size: {}'.format(len(train_indices)))
    logging.info('Validation size: {}'.format(len(val_indices)))

    # Creating data samplers and loaders
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    train_data = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=train_sampler,
        pin_memory=True
    )
    val_data = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        sampler=val_sampler
    )
    logging.info('Done')

    # Load the network
    logging.info('Loading Network...')
    input_channels = 1 * args.use_depth + 3 * args.use_rgb
    logging.info("input channel is {}".format(input_channels))
    network = get_network(args.network)
    net = network(
        input_channels=input_channels,
        dropout=args.use_dropout,
        prob=args.dropout_prob,
        channel_size=args.channel_size,
        upsamp=args.upsamp,
        att = args.att,
        use_mish=args.use_mish
    )
    if args.goon_train:
        # 加载预训练模型
        net = torch.load(args.model, map_location=torch.device(device))
        # net.load_state_dict(pretrained_dict, strict=True)   # True:完全吻合，False:只加载键值相同的参数，其他加载默认值。
        logging.info('Done from goon checkpoint >>>{}'.format(args.model))
    else :
        logging.info('Done from zero model')
    net = net.to(device)
    
    base_lr = args.lr
    if args.optim.lower() == 'adam':
        # 优化器
        optimizer = optim.Adam(net.parameters(), lr=base_lr, weight_decay=args.weight_decay)
    elif args.optim.lower() == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)
    elif args.optim.lower() == 'ranger':
        base_lr = 1e-4
        optimizer = Ranger(net.parameters(), lr=base_lr)
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(args.optim))
    logging.info('optimizer {} Done'.format(args.optim))

    total_epochs = args.epochs
    epoch_len = args.batches_per_epoch
    total_iters = epoch_len * total_epochs // 2

    if args.scheduler.lower() == 'flat':
        scheduler = flat_and_anneal_lr_scheduler(
        optimizer=optimizer,
        total_iters=total_iters,
        warmup_method="linear",
        warmup_factor=0.1,
        warmup_iters=800,
        anneal_method="cosine",
        anneal_point=0.72,
        target_lr_factor=0.0,
        poly_power=5,
        step_gamma=0.1,
        steps=[0.5, 0.75, 0.9],
        )
    elif args.scheduler.lower() == 'multi_step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,15,25,35], gamma=0.5)
    else :
        logging.info('scheduler para error check')

    logging.info('scheduler {} Done'.format(args.optim))
    logging.info("start lr: {}".format(scheduler.get_lr()))
    # Print model architecture.
    summary(net, (input_channels, args.input_size, args.input_size))
    f = open(os.path.join(save_folder, 'arch.txt'), 'w')
    sys.stdout = f
    summary(net, (input_channels, args.input_size, args.input_size))
    sys.stdout = sys.__stdout__
    f.close()

    best_iou = 0.0
    start_epoch = args.start_epoch if args.goon_train else 0
    for _ in range(start_epoch):
        for _ in range(epoch_len):
            scheduler.step() # when no state_dict availble
            global_step += 1

    for epoch in range(start_epoch,total_epochs):
        logging.info('Beginning Epoch {:02d}, lr={}'.format(epoch, scheduler.get_lr()[0]))
        epoch_lrs.append([epoch, scheduler.get_lr()[0]])  # only get the first lr (maybe a group of lrs)
        train_results = train(epoch, net, device, train_data, scheduler, epoch_len, vis=args.vis)
        # # Log training losses to tensorboard
        # tb.add_scalar('loss/train_loss', train_results['loss'], epoch)
        # for n, l in train_results['losses'].items():
        #     tb.add_scalar('train_loss/' + n, l, epoch)

        # if args.iou_abla == True:
        #     logging.info('Validating 0.40...')
        #     test_results = validate(net, device, val_data, 0.40)
        #     logging.info('%d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
        #                                 test_results['correct'] / (test_results['correct'] + test_results['failed'])))
        #     logging.info('Validating 0.35...')
        #     test_results = validate(net, device, val_data, 0.35)
        #     logging.info('%d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
        #                                 test_results['correct'] / (test_results['correct'] + test_results['failed'])))
        #     logging.info('Validating 0.30...')
        #     test_results = validate(net, device, val_data, 0.30)
        #     logging.info('%d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
        #                                 test_results['correct'] / (test_results['correct'] + test_results['failed'])))

        # # Run Validation
        # logging.info('Validating 0.25...')
        # test_results = validate(net, device, val_data, 0.25)
        # logging.info('%d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
        #                              test_results['correct'] / (test_results['correct'] + test_results['failed'])))
        # # Log validation results to tensorbaord
        # tb.add_scalar('loss/IOU', test_results['correct'] / (test_results['correct'] + test_results['failed']), epoch)
        # tb.add_scalar('loss/val_loss', test_results['loss'], epoch)
        # for n, l in test_results['losses'].items():
        #     tb.add_scalar('val_loss/' + n, l, epoch)

        # # Save best performing network
        # iou = test_results['correct'] / (test_results['correct'] + test_results['failed'])
        # if iou > best_iou or epoch == 0 or (epoch % 10) == 0:
        #     logging.info('>>> save model: epoch_%02d_iou_%0.4f' % (epoch, iou))
        #     # torch.save(net.state_dict(), os.path.join(save_folder, 'epoch_%02d_iou_%0.4f' % (epoch, iou)))
        #     torch.save(net, os.path.join(save_folder, 'epoch_%02d_iou_%0.4f' % (epoch, iou)))
        #     best_iou = iou

    if args.vis_lr:
        import matplotlib.pyplot as plt
        epoch_lrs = np.asarray(epoch_lrs, dtype=np.float32)
        for i in range(len(epoch_lrs)):
            print("{:02d} {}".format(int(epoch_lrs[i][0]), epoch_lrs[i][1]))

        plt.figure(dpi=100)
        plt.suptitle("learning rate curve")
        plt.subplot(1, 2, 1)
        plt.plot(steps, lrs, "-.")
        # plt.show()
        plt.subplot(1, 2, 2)
        # print(epoch_lrs.dtype)
        plt.plot(epoch_lrs[:, 0], epoch_lrs[:, 1], "-.")
        plt.show()

if __name__ == '__main__':
    run()
