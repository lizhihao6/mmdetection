_base_ = ['../_base_/models/yolo_rgb.py']
model = dict(bbox_head=dict(num_classes=1))
# dataset settings
dataset_type = 'CityscapesDataset'
data_root = '/shared/cityscapes/'
img_norm_cfg = dict(
    mean=[528, 528, 528], std=[4095 - 528, 4095 - 528, 4095 - 528], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize', img_scale=(1024, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=16,
    train=dict(
        type='RepeatDataset',
        times=8,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root +
                     'mmdetection_labels/instancesonly_filtered_gtFine_train.json',
            img_prefix=data_root + 'CycleISP/train/',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root +
                 'mmdetection_labels/instancesonly_filtered_gtFine_val.json',
        img_prefix=data_root + 'CycleISP/val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root +
                 'mmdetection_labels/instancesonly_filtered_gtFine_test.json',
        img_prefix=data_root + 'CycleISP/test/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
