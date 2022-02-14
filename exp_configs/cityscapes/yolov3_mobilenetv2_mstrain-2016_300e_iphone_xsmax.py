_base_ = ['../_base_/models/yolo_rgb.py', '../_base_/datasets/iphone_xsmax.py']
dataset_type = 'MultiRAWCityscapesDataset'
model = dict(bbox_head=dict(num_classes=8))

img_norm_cfg = dict(mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
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
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    val=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        rearrange_bbox_at_test=False),
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        rearrange_bbox_at_test=False))
evaluation = dict(interval=1, metric='mAP')
