import os
import time
import argparse
import logging
import numpy as np
import torch
from itertools import chain

from timm.models import create_model, apply_test_time_pool
from timm.data import Dataset, create_loader, resolve_data_config
from timm.utils import AverageMeter, setup_default_logging

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('inference')


parser = argparse.ArgumentParser(description='PyTorch Inference')
parser.add_argument('-d', '--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--output_dir', metavar='DIR', default='./',
                    help='path to output files')
parser.add_argument('--model', '-m', metavar='MODEL', default='tf_efficientnet_b7',
                    help='model architecture (default: tf_efficientnet_b7)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num_classes', type=int, default=1000,
                    help='Number classes in dataset')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num_gpu', type=int, default=7,
                    help='Number of GPUS to use')
parser.add_argument('--no-test-pool', dest='no_test_pool', action='store_true',
                    help='disable test time pool')
parser.add_argument('--topk', default=5, type=int,
                    metavar='N', help='Top-k to output to CSV')

args = parser.parse_args()

def predict(INP_DIR, BATCH_SIZE, NUMS_CLASS, MODEL_NAME, MODEL_PATH, OUT_DIR):
    print("[INFO] Predicting")
    setup_default_logging()
    args = parser.parse_args()
    # might as well try to do something useful...

    args.pretrained = args.pretrained or not MODEL_PATH
    
    # create model
    model = create_model(
        MODEL_NAME,
        num_classes=NUMS_CLASS,
        in_chans=3,
        pretrained=args.pretrained,
        checkpoint_path=MODEL_PATH)

    _logger.info('Model %s created, param count: %d' %
                 (args.model, sum([m.numel() for m in model.parameters()])))

    config = resolve_data_config(vars(args), model=model)
    model, test_time_pool = (model, False) if args.no_test_pool else apply_test_time_pool(model, config)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
    else:
        model = model.cuda()

    loader = create_loader(
        Dataset(INP_DIR),
        input_size=config['input_size'],
        batch_size=BATCH_SIZE,
        use_prefetcher=True,
        interpolation=config['interpolation'],
        mean=config['mean'],
        std=config['std'],
        num_workers=args.workers,
        crop_pct=1.0 if test_time_pool else config['crop_pct'])

    model.eval()

    k = min(args.topk, args.num_classes)
    batch_time = AverageMeter()
    end = time.time()
    topk_ids = []
    topk_prob = []
    with torch.no_grad():
        for batch_idx, (input, _) in enumerate(loader):
            try:
                input = input.cuda()
                labels = model(input)
                topk = labels.topk(k)[0]
                topk = topk.cpu().numpy()
                # print(topk)
                topk = np.exp(topk) / np.sum(np.exp(topk), axis=-1)[:, np.newaxis]
                # print(topk)
                topk_prob.append(topk)
                topk = labels.topk(k)[1]
                topk_ids.append(topk.cpu().numpy())

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if batch_idx % args.log_freq == 0:
                    _logger.info('Predict: [{0}/{1}] Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                        batch_idx, len(loader), batch_time=batch_time))
            except Exception:
                pass

    topk_ids = np.concatenate(topk_ids, axis=0).squeeze()
    topk_prob = np.concatenate(topk_prob, axis=0).squeeze()
    
    print("topk_ids: ", topk_ids)   
    print("topk_prob: ", topk_prob)

    # out_path = os.path.join(args.output_dir, 'submission_{}.txt'.format(args.model))
    out_path = os.path.join(OUT_DIR, "output_single_class_stage1.txt")
    with open(out_path, 'w') as out_file:
        filenames = loader.dataset.filenames(basename=True)
        for filename, label, prob in zip(filenames, topk_ids, topk_prob):
            # out_file.write(("{}" + "\t{}\t{:.4f}"*5 + "\n").format(
            #     filename, *chain(*zip(label, prob))))
            out_file.write(("{}" + " {} {:.4f}"*5 + "\n").format(
                filename, *chain(*zip(label, prob))))
            
            print("label: {}, prob: {}".format(label, prob))
    
    return out_path
    # return topk_ids

if __name__ == '__main__':
    predict(args.data, args.batch_size, args.num_classes, args.model, args.output_dir)