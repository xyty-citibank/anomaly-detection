
model = dict(
    type='',
    pretrained='',
    flownet=dict(),
    apcnet=dict(

    ),


)
train_cfg = dict(
    lambda_int=None,
    lambda_gd=None,
    lambda_op=None,
    lambda_adv=None,
    alpha=None,
    ganmodel=None,
    lr=None,
    start_epoch=0,


)

test_cfg = dict(

)

dataset_type = ''
data_root = ''
img_norm_cfg = dict()

data = dict(
    videos_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file='',
        img_prefix=data_root,
        img_norm_cfg=img_norm_cfg,
        image_tmpl='img_{:05d}.jpg',
        img_scale=256,
        input_size=224,
    ),
    val=dict(),
    test=dict(),
)


#optimizer
optimizer = dict(type='SGD', lr=0.02)

lr_config = dict()

checkpoint_config = dict(
    interval=10,
)
log_config = dict(
    interval=20,
)

total_epochs = 50

work_dir = ''

resum_from = None