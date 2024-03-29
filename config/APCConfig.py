
model = dict(
    type=None,
    pretrained=None,
    flownet=dict(),
    apcnet=dict(


    ),
    gan=dict(
        ganmodel='lsgan',

    ),



)
train_cfg = dict(
    lambda_int=1.0,
    lambda_gd=1.0,
    lambda_op=1.0,
    lambda_adv=1.0,
    alpha=1,
    flownet_pretrained='/ssd/project/anomaly-detection/mmanomaly/FlowNetPytorch/checkpoint/flownets_EPE1.951.pth.tar',



)

test_cfg = dict(

)

dataset_type = 'subway'
# data_root = '/vdata/dataset/Avenue_Dataset/training_rawframes'
data_root = '/vdata/dataset/events/subway_entrance/train/train_rawframes'
img_norm_cfg = dict(
   mean=(127.5, 127.5, 127.5), std=(127.5, 127.5, 127.5), to_rgb=True)
data = dict(
    videos_per_gpu=6,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=None,
        v_prefix=data_root,
        scale_size=(3, 256, 256),
        num_pred=1,
        time_steps=4
    ),
    val=dict(
        type=dataset_type,
        ann_file=None,
        v_prefix=data_root,
        scale_size=(3, 256, 256),
        num_pred=1,
        time_steps=4
    ),
    test=dict(
        type=dataset_type,
        ann_file=None,
        v_prefix=data_root,
        scale_size=(3, 256, 256),
        num_pred=1,
        time_steps=4
    ),
)


#optimizer
optimizer = [dict(type='Adam', lr=0.0002), dict(type='Adam', lr=0.0002)]
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup=None,
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[20000, 100000])

checkpoint_config = dict(
    interval=1000,
)

log_config = dict(
    interval=1,
    hooks=[
        dict(type='MyTextLoggerHook'),
    ])
total_epochs = 200000

work_dir = './work_dirs/ano_pred_cvpr2018_model_subway_entrance'
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]