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
import tqdm
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
    parser.add_argument('--iou-abla', type=bool, default=True,
                        help='Threshold albation for evaluation, need more time')
    
    parser.add_argument('--use-mish', type=bool, default=True,
                        help='(  True  False  )')
    parser.add_argument('--posloss', type=bool, default=True,
                        help='(  True  False  )')
    parser.add_argument('--upsamp', type=str, default='use_bilinear',
                        help='Use upsamp type (  use_duc  use_convt use_bilinear use_nearest  )')
    parser.add_argument('--att', type=str, default='use_coora',
                        help='Use att type (  use_eca  use_se use_coora use_cba)')
    parser.add_argument('--use_gauss_kernel', type=float, default= 0.0,
                        help='Dataset gaussian progress 0.0 means not use gauss')
    parser.add_argument('--datarotate', type=bool, default=True,
                        help='Threshold albation for evaluation, need more time')
    parser.add_argument('--datazoom', type=bool, default=True,
                        help='Threshold albation for evaluation, need more time')

    # /media/lab/ChainGOAT/Jacquard
    # /media/lab/e/zzy/datasets/Cornell
    parser.add_argument('--dataset', type=str,default='jacquard',
                        help='Dataset Name ("cornell" or "jacquard"  graspnet1b )')
    parser.add_argument('--dataset-path', type=str,default='/media/lab/ChainGOAT/Jacquard',
                        help='Path to dataset')
    parser.add_argument('--alfa', type=int, default=1,
                        help='len(Dataset)*alfa')
    parser.add_argument('--split', type=float, default=0.9,
                        help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-shuffle', action='store_true', default=False,
                        help='Shuffle the dataset')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split')
    parser.add_argument('--num-workers', type=int, default=18,
                        help='Dataset workers')

    # Training
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    
    parser.add_argument('--optim', type=str, default='ranger',
                        help='Optmizer for the training. (adam or SGD)')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--schedu-milestone', type=int, default=[5,15,25,35], help='学习率tiaozheng stone')
    parser.add_argument('--schedu-gamma', type=float, default=0.5, help='学习率 hsuaijian xishu ')
    parser.add_argument('--weight-decay', type=float, default=0, help='权重衰减 L2正则化系数')

    parser.add_argument('--epochs', type=int, default=60,
                        help='Training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=1600,
                        help='Batches per Epoch')
    
    parser.add_argument('--goon-train', type=bool, default=False, help='是否从已有网络继续训练')
    parser.add_argument('--model', type=str, default='logs/jacquard_ftn/230226_1444_dwc1_new_drop2_ga0/epoch_01_iou_0.8811', help='保存的模型')
    parser.add_argument('--start-epoch', type=int, default=2, help='继续训练开始的epoch')
    

    # Logging etc.
    parser.add_argument('--description', type=str, default='dwc1_322_drop2_ga0_gama',
                        help='Training description')
    parser.add_argument('--logdir', type=str, default='logs/jacquard_ftn',
                        help='Log directory')
    parser.add_argument('--vis', action='store_true',
                        help='Visualise the training process')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')
    parser.add_argument('--random-seed', type=int, default=123,
                        help='Random seed for numpy')
    args = parser.parse_args()
    return args


def validate(net, device, val_data, iou_multi=False,posloss=True):
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
        'correct': {
        'th25':0,
        'th30':0,
        'th35':0,
        'th40':0,
        },
        'failed': {
        'th25':0,
        'th30':0,
        'th35':0,
        'th40':0,
        },
        'loss': 0,
        'losses': {
        }
    }

    ld = len(val_data)

    with torch.no_grad():
        with tqdm.tqdm(total=val_data.__len__(),ncols=70,colour='blue') as pb:
            for x, y, didx, rot, zoom_factor in val_data:
                xc = x.to(device)
                yc = [yy.to(device) for yy in y]
                lossd = net.compute_loss(xc, yc,pos_loss=posloss)

                loss = lossd['loss']

                results['loss'] += loss.item() / ld
                for ln, l in lossd['losses'].items():
                    if ln not in results['losses']:
                        results['losses'][ln] = 0
                    results['losses'][ln] += l.item() / ld

                q_out, ang_out, w_out = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                            lossd['pred']['sin'], lossd['pred']['width'])


                if iou_multi:
                    iou_results = evaluation.calculate_iou_match_multi(q_out,
                                                    ang_out,
                                                    val_data.dataset.get_gtbb(didx, rot, zoom_factor),
                                                    no_grasps=1,
                                                    grasp_width=w_out
                                                    )

                    for iou in iou_results:
                        if iou_results[iou] == True:
                            results['correct'][iou] +=1
                        else :
                            results['failed'][iou] +=1
                else :
                    s = evaluation.calculate_iou_match(q_out, ang_out,
                                                    val_data.dataset.get_gtbb(didx, rot, zoom_factor),
                                                    no_grasps=1,
                                                    grasp_width=w_out,
                                                    )

                    if s:
                        results['correct']['th25'] += 1
                    else:
                        results['failed']['th25'] += 1
                pb.set_postfix(loss='{:0.4f}'.format(loss.item()))
                pb.update(1)

    return results


def train(epoch, net, device, train_data, optimizer, batches_per_epoch, vis=False,posloss=True):
    """
    Run one training epoch
    :param epoch: Current epoch
    :param net: Network
    :param device: Torch device
    :param train_data: Training Dataset
    :param optimizer: Optimizer
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
    # Use batches per epoch to make training on different sized datasets (cornell/jacquard) more equivalent.
    with tqdm.tqdm(total=batches_per_epoch,ncols=70,colour='yellow') as pb:
        while batch_idx <= batches_per_epoch:
            for x, y, _, _, _ in train_data:
                batch_idx += 1
                if batch_idx > batches_per_epoch:
                    break

                xc = x.to(device)
                yc = [yy.to(device) for yy in y]
                lossd = net.compute_loss(xc, yc,pos_loss=posloss)

                loss = lossd['loss']

                # if batch_idx % 100 == 0:
                #     logging.info('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss.item()))

                results['loss'] += loss.item()
                for ln, l in lossd['losses'].items():
                    if ln not in results['losses']:
                        results['losses'][ln] = 0
                    results['losses'][ln] += l.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Display the images
                if vis:
                    imgs = []
                    n_img = min(4, x.shape[0])
                    for idx in range(n_img):
                        imgs.extend([x[idx,].numpy().squeeze()] + [yi[idx,].numpy().squeeze() for yi in y] + [
                            x[idx,].numpy().squeeze()] + [pc[idx,].detach().cpu().numpy().squeeze() for pc in
                                                        lossd['pred'].values()])
                    gridshow('Display', imgs,
                            [(xc.min().item(), xc.max().item()), (0.0, 1.0), (0.0, 1.0), (-1.0, 1.0),
                            (0.0, 1.0)] * 2 * n_img,
                            [cv2.COLORMAP_BONE] * 10 * n_img, 10)
                    cv2.waitKey(2)
                pb.update(1)
                pb.set_postfix(loss='{:0.4f}'.format(loss.item()))

    results['loss'] /= batch_idx
    for l in results['losses']:
        results['losses'][l] /= batch_idx

    return results


def run():
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
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-2s: %(levelname)-2s %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    # Get the compute device
    device = get_device(args.force_cpu)

    # Load Dataset
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)
    dataset = Dataset(args.dataset_path,
                      output_size=args.input_size,
                      ds_rotate=args.ds_rotate,
                      alfa=args.alfa,
                      random_rotate=args.datarotate,
                      random_zoom=args.datazoom,
                      include_depth=args.use_depth,
                      include_rgb=args.use_rgb,
                      use_gauss_kernel = args.use_gauss_kernel)
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
        logging.info('Done from goon checkpoint {}'.format(args.model))
    else :
        logging.info('Done from zero model')
    net = net.to(device)
    

    if args.optim.lower() == 'adam':
        # 优化器
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedu_milestone, gamma=args.schedu_gamma)     # 学习率衰减    20, 30, 60
    elif args.optim.lower() == 'adamw':
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedu_milestone, gamma=args.schedu_gamma)     # 学习率衰减    20, 30, 60
    elif args.optim.lower() == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedu_milestone, gamma=args.schedu_gamma)     # 学习率衰减    20, 30, 60
    elif args.optim.lower() == 'ranger':
        optimizer = Ranger(net.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedu_milestone, gamma=args.schedu_gamma)
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(args.optim))
    
    # Print model architecture.
    summary(net, (input_channels, args.input_size, args.input_size))
    f = open(os.path.join(save_folder, 'arch.txt'), 'w')
    sys.stdout = f
    summary(net, (input_channels, args.input_size, args.input_size))
    sys.stdout = sys.__stdout__
    f.close()

    logging.info('Optimizer {} Done || Base LR: {} || Milestone: {} || Gamma: {}'.format(args.optim,args.lr,args.schedu_milestone,args.schedu_gamma))
    logging.info('Dataset size {}   ||rotate :{}   || zoom :{}      || ds_shuffle {} || ds_rotate {}'.format(dataset.length,args.datarotate,args.datazoom,args.ds_shuffle,args.ds_rotate))

    best_iou = 0.0
    start_epoch = args.start_epoch if args.goon_train else 0
    for _ in range(start_epoch):
        scheduler.step()
    for epoch in range(args.epochs)[start_epoch:]:
        logging.info('=================Beginning Epoch {:02d}, lr={}==============='.format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
        if args.dataset.title() != 'Jacquard':
            train_results = train(epoch, net, device, train_data, optimizer, train_data.__len__()*10, vis=args.vis,posloss=args.posloss)
        else :
            train_results = train(epoch, net, device, train_data, optimizer, train_data.__len__(), vis=args.vis,posloss=args.posloss)
        logging.info("====== Epoch {:02d} train loss {:0.4f} ========".format(epoch,train_results['loss']))
        scheduler.step()
        # Log training losses to tensorboard
        tb.add_scalar('loss/train_loss', train_results['loss'], epoch)
        for n, l in train_results['losses'].items():
            tb.add_scalar('train_loss/' + n, l, epoch)

        # Run Validation
        logging.info('Validating ...')
        test_results = validate(net, device, val_data,  iou_multi=args.iou_abla,posloss=args.posloss)
        logging.info("====== Epoch {:02d} Vali loss {:0.4f} ========".format(epoch,test_results['loss']))
        if args.iou_abla:
            for result in test_results['correct']:
                logging.info('+++++ Epoch %d Jacquard index %s  %d/%d = %f' % (epoch,result,test_results['correct'][result], test_results['correct'][result] + test_results['failed'][result],
                                        test_results['correct'][result] / (test_results['correct'][result] + test_results['failed'][result])))
        else :
            logging.info('+++++ Epoch %d Validating IOU %d/%d = %f' % (epoch,test_results['correct']['th25'], test_results['correct']['th25'] + test_results['failed']['th25'],
                                     test_results['correct']['th25']/(test_results['correct']['th25']+test_results['failed']['th25'])))
        # Log validation results to tensorbaord
        tb.add_scalar('loss/IOU', test_results['correct']['th25'] / (test_results['correct']['th25'] + test_results['failed']['th25']), epoch)
        tb.add_scalar('loss/val_loss', test_results['loss'], epoch)
        for n, l in test_results['losses'].items():
            tb.add_scalar('val_loss/' + n, l, epoch)

        # Save best performing network
        iou = test_results['correct']['th25'] / (test_results['correct']['th25'] + test_results['failed']['th25'])
        if iou > best_iou or epoch == 0 or (epoch % 10) == 0:
            logging.info('>>>>>>>>>> save model: epoch_%02d_iou_%0.4f' % (epoch, iou))
            torch.save(net, os.path.join(save_folder, 'epoch_%02d_iou_%0.4f' % (epoch, iou)))
            best_iou = iou

if __name__ == '__main__':
    run()
