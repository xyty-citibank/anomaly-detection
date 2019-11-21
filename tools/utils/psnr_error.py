
import torch


def psnr_error(gen_frames, gt_frames):

    gt_frames = (gt_frames + 1.0) / 2.0
    gen_frames = (gen_frames + 1.0) / 2.0
    square_diff = torch.sqrt(gt_frames - gen_frames)
    shape = gen_frames.shape
    num_pixels = shape[0] * shape[1] * shape[2]
    batch_errors = 10 * torch.log10(1 / ((1 / num_pixels) * torch.sum(square_diff)))
    return batch_errors







