
import pickle
import argparse
import os
import torch
import mmcv
from  tools import evaluate
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict
from mmanomaly.models.ano_pred_cvpr2018_model import APCModel
from mmcv.parallel import scatter, collate, MMDataParallel
from mmanomaly.datasets.transform import GroupImageTransform
from mmanomaly.datasets.utils import *
from tools.utils.psnr_error import psnr_error
from mmcv import Config

def parse_args():
    parser = argparse.ArgumentParser(description='Test an action recognizer')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--use_softmax', action='store_true',
                        help='whether to use softmax score')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.gpus = args.gpus

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    model = APCModel(cfg)
    load_checkpoint(model, args.checkpoint, strict=True)
    generater = model.generator
    generater = MMDataParallel(generater, device_ids=[0])
    generater.eval()

    transform = GroupImageTransform(mean=cfg.img_norm_cfg['mean'], std=cfg.img_norm_cfg['std'], to_rgb=cfg.img_norm_cfg['to_rgb'])

    video_path = cfg.data.test.v_prefix
    video_list = os.listdir(video_path)
    video_list.sort()

    psnr_records = []
    for video in video_list:
        frame_list = os.listdir(os.path.join(video_path, video))
        frame_list.sort()
        size = len(frame_list)
        clip_length = cfg.data.test.time_steps + cfg.data.test.num_pred
        c, w, h = cfg.data.val.scale_size
        psnrs = np.empty(shape=(size,), dtype=np.float32)
        for i in range(clip_length, size - 1):
            frame_clip = frame_list[i - clip_length:i]
            frame_arr = []
            for frame_path in frame_clip:
                frame_arr.append(mmcv.imread(os.path.join(video_path, video, frame_path)))

            frames, img_shape, pad_shape, scale_factor, crop_quadruple = transform(frame_arr, (w, h),
                                                                                        keep_ratio=False,
                                                                                        div_255=False)
            frames = to_tensor(frames)
            size = len(frame_clip)
            frames = frames.reshape(c * size, w, h).unsqueeze(0)
            g_t = frames[:, :c * cfg.data.test.time_steps, :, :]
            g_t_1 = frames[:, c * cfg.data.test.time_steps:, :, :]
            p_t_1 = generater(g_t)

            psnr = psnr_error(p_t_1, g_t_1.cuda())
            psnrs[i] = psnr
        psnrs[:clip_length] = psnrs[clip_length]
        psnr_records.append(psnrs)

    result_dict = {'dataset': 'avenue', 'psnr': psnr_records, 'flow': [], 'names': [], 'diff_mask': []}



    # # TODO specify what's the actual name of ckpt.
    pickle_path = '../result/Avenue.pkl'
    with open(pickle_path, 'wb') as writer:
        pickle.dump(result_dict, writer, pickle.HIGHEST_PROTOCOL)

    results = evaluate.evaluate('compute_auc', pickle_path)
    print(results)


if __name__ == '__main__':
    main()





























