


import argparse
import sys
import os
import os.path as osp
import glob
from multiprocessing import Pool

import mmcv


def dump_frames(vid_item):
    full_path, vid_path, vid_id = vid_item
    vid_name = vid_path.split('.')[0]
    out_full_path = osp.join(args.out_dir, vid_name)
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass
    vr = mmcv.VideoReader(full_path)
    for i in range(len(vr)):
        if vr[i] is not None:
            mmcv.imwrite(
                vr[i], '{}/img_{:05d}.jpg'.format(out_full_path, i + 1))
        else:
            print('[Warning] length inconsistent!'
                  'Early stop with {} out of {} frames'.format(i + 1, len(vr)))
            break
    print('{} done with {} frames'.format(vid_name, len(vr)))
    sys.stdout.flush()
    return True



def parse_args():
    parser = argparse.ArgumentParser(description='extract optical flows')
    parser.add_argument('src_dir', type=str)
    parser.add_argument('out_dir', type=str)
    parser.add_argument('--level', type=int,
                        choices=[1, 2],
                        default=2)
    parser.add_argument('--num_worker', type=int, default=8)

    parser.add_argument("--out_format", type=str, default='dir',
                        choices=['dir', 'zip'], help='output format')
    parser.add_argument("--ext", type=str, default='avi',
                        choices=['avi', 'mp4'], help='video file extensions')
    parser.add_argument("--new_width", type=int, default=0,
                        help='resize image width')
    parser.add_argument("--new_height", type=int,
                        default=0, help='resize image height')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not osp.isdir(args.out_dir):
        print('Creating folder: {}'.format(args.out_dir))
        os.makedirs(args.out_dir)
    if args.level == 2:
        classes = os.listdir(args.src_dir)
        for classname in classes:
            new_dir = osp.join(args.out_dir, classname.split('.')[0])
            if not osp.isdir(new_dir):
                print('Creating folder: {}'.format(new_dir))
                os.makedirs(new_dir)

    print('Reading videos from folder: ', args.src_dir)
    print('Extension of videos: ', args.ext)
    if args.level == 2:
        fullpath_list = glob.glob(args.src_dir + '/*/*.' + args.ext)
        done_fullpath_list = glob.glob(args.out_dir + '/*/*')
    elif args.level == 1:
        fullpath_list = glob.glob(args.src_dir + '/*.' + args.ext)
        done_fullpath_list = glob.glob(args.out_dir + '/*')
    print('Total number of videos found: ', len(fullpath_list))

    if args.level == 2:
        vid_list = list(map(lambda p: osp.join(
            '/'.join(p.split('/')[-2:])), fullpath_list))
    elif args.level == 1:
        vid_list = list(map(lambda p: p.split('/')[-1], fullpath_list))

    pool = Pool(args.num_worker)
    pool.map(dump_frames, zip(
            fullpath_list, vid_list, range(len(vid_list))))
