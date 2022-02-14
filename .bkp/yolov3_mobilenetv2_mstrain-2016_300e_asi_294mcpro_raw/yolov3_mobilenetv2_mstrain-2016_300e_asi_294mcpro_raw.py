checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
model = dict(
    type='YOLOV3',
    backbone=dict(
        type='MobileNetV2RAW',
        in_ch=8,
        use_resizer=False,
        out_indices=(2, 4, 6),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        init_cfg=None),
    neck=dict(
        type='YOLOV3Neck',
        num_scales=3,
        in_channels=[320, 96, 32],
        out_channels=[96, 96, 96]),
    bbox_head=dict(
        type='YOLOV3Head',
        num_classes=4,
        in_channels=[96, 96, 96],
        out_channels=[96, 96, 96],
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[[(116, 90), (156, 198), (373, 326)],
                        [(30, 61), (62, 45), (59, 119)],
                        [(10, 13), (16, 30), (33, 23)]],
            strides=[32, 16, 8]),
        bbox_coder=dict(type='YOLOBBoxCoder'),
        featmap_strides=[32, 16, 8],
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_conf=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_xy=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=2.0,
            reduction='sum'),
        loss_wh=dict(type='MSELoss', loss_weight=2.0, reduction='sum')),
    train_cfg=dict(
        assigner=dict(
            type='GridAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0)),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        conf_thr=0.005,
        nms=dict(type='nms', iou_threshold=0.45),
        max_per_img=100))
optimizer = dict(type='SGD', lr=0.003, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=4000,
    warmup_ratio=0.0001,
    step=[24, 28])
runner = dict(type='EpochBasedRunner', max_epochs=30)
find_unused_parameters = True
dataset_type = 'MultiRAWDataset'
img_norm_cfg = dict(
    mean=[0, 0, 0, 0, 0, 0, 0, 0], std=[1, 1, 1, 1, 1, 1, 1, 1], to_rgb=False)
train_pipeline = [
    dict(type='LoadRAWFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Rearrange'),
    dict(type='HDRSplit'),
    dict(
        type='Expand',
        mean=[0, 0, 0, 0, 0, 0, 0, 0],
        to_rgb=False,
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(2016, 1512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[0, 0, 0, 0, 0, 0, 0, 0],
        std=[1, 1, 1, 1, 1, 1, 1, 1],
        to_rgb=False),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadRAWFromFile'),
    dict(type='Rearrange'),
    dict(type='HDRSplit'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2016, 1512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[0, 0, 0, 0, 0, 0, 0, 0],
                std=[1, 1, 1, 1, 1, 1, 1, 1],
                to_rgb=False),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=10,
        dataset=dict(
            type='MultiRAWDataset',
            img_subdir='raw',
            meta_subdir='metainfo',
            ann_subdir='labels/detection',
            pipeline=[
                dict(type='LoadRAWFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='Rearrange'),
                dict(type='HDRSplit'),
                dict(
                    type='Expand',
                    mean=[0, 0, 0, 0, 0, 0, 0, 0],
                    to_rgb=False,
                    ratio_range=(1, 2)),
                dict(
                    type='MinIoURandomCrop',
                    min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                    min_crop_size=0.3),
                dict(type='Resize', img_scale=(2016, 1512), keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Normalize',
                    mean=[0, 0, 0, 0, 0, 0, 0, 0],
                    std=[1, 1, 1, 1, 1, 1, 1, 1],
                    to_rgb=False),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            ],
            rearrange_bbox_at_test=True,
            ann_file='/lzh/datasets/multiRAW/asi_294mcpro/train.txt',
            img_prefix='/lzh/datasets/multiRAW/asi_294mcpro/',
            img_suffix='raw.TIF')),
    val=dict(
        type='MultiRAWDataset',
        img_subdir='raw',
        meta_subdir='metainfo',
        ann_subdir='labels/detection',
        pipeline=[
            dict(type='LoadRAWFromFile'),
            dict(type='Rearrange'),
            dict(type='HDRSplit'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2016, 1512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[0, 0, 0, 0, 0, 0, 0, 0],
                        std=[1, 1, 1, 1, 1, 1, 1, 1],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        rearrange_bbox_at_test=True,
        ann_file='/lzh/datasets/multiRAW/asi_294mcpro/test.txt',
        img_prefix='/lzh/datasets/multiRAW/asi_294mcpro/',
        img_suffix='raw.TIF'),
    test=dict(
        type='MultiRAWDataset',
        img_subdir='raw',
        meta_subdir='metainfo',
        ann_subdir='labels/detection',
        pipeline=[
            dict(type='LoadRAWFromFile'),
            dict(type='Rearrange'),
            dict(type='HDRSplit'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2016, 1512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[0, 0, 0, 0, 0, 0, 0, 0],
                        std=[1, 1, 1, 1, 1, 1, 1, 1],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        rearrange_bbox_at_test=True,
        ann_file='/lzh/datasets/multiRAW/asi_294mcpro/test.txt',
        img_prefix='/lzh/datasets/multiRAW/asi_294mcpro/',
        img_suffix='raw.TIF'))
evaluation = dict(interval=1, metric='mAP')
data_root = '/lzh/datasets/multiRAW/asi_294mcpro/'
suffix = 'raw.TIF'
work_dir = './work_dirs/yolov3_mobilenetv2_mstrain-2016_300e_asi_294mcpro_raw'
gpu_ids = range(0, 4)
