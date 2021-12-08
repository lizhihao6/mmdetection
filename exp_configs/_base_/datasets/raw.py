# dataset settings
dataset_type = 'MultiRAWDataset'
img_norm_cfg = dict(mean=[0 for _ in range(8)], std=[1 for _ in range(8)], to_rgb=False)
train_pipeline = [
    dict(type='LoadRAWFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Rearrange'),
    dict(type='HDRSplit'),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(2016, 1512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
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
            dict(type='Normalize', **img_norm_cfg),
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
            type=dataset_type,
            img_subdir='raw',
            meta_subdir='metainfo',
            ann_subdir='labels/detection',
            pipeline=train_pipeline,
            rearrange_bbox_at_test=True)),
    val=dict(
        type=dataset_type,
        img_subdir='raw',
        meta_subdir='metainfo',
        ann_subdir='labels/detection',
        pipeline=test_pipeline,
        rearrange_bbox_at_test=True),
    test=dict(
        type=dataset_type,
        img_subdir='raw',
        meta_subdir='metainfo',
        ann_subdir='labels/detection',
        pipeline=test_pipeline,
        rearrange_bbox_at_test=True))
evaluation = dict(interval=1, metric='mAP')
