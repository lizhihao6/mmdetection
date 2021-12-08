checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = 'work_dirs/yolov3_mobilenetv2_mstrain-2016_300e_huawei_p30pro/latest.pth'
workflow = [('train', 1)]
dataset_type = 'MultiRAWDataset'
data_root = '/lzh/datasets/multiRAW/huawei_p30pro/'
img_norm_cfg = dict(mean=[0, 0, 0], std=[255.0, 255.0, 255.0], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Expand', mean=[0, 0, 0], to_rgb=True, ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(2016, 1512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[0, 0, 0],
        std=[255.0, 255.0, 255.0],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2016, 1512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[0, 0, 0],
                std=[255.0, 255.0, 255.0],
                to_rgb=True),
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
            ann_file='/lzh/datasets/multiRAW/huawei_p30pro/train.txt',
            img_prefix='/lzh/datasets/multiRAW/huawei_p30pro/',
            img_subdir='camera_isp',
            meta_subdir='place_holder',
            ann_subdir='labels/detection',
            img_suffix='jpg',
            pipeline=[
                dict(type='LoadImageFromFile', to_float32=True),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(
                    type='Expand',
                    mean=[0, 0, 0],
                    to_rgb=True,
                    ratio_range=(1, 2)),
                dict(
                    type='MinIoURandomCrop',
                    min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                    min_crop_size=0.3),
                dict(type='Resize', img_scale=(2016, 1512), keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Normalize',
                    mean=[0, 0, 0],
                    std=[255.0, 255.0, 255.0],
                    to_rgb=True),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            ],
            rearrange_bbox_at_test=False)),
    val=dict(
        type='MultiRAWDataset',
        ann_file='/lzh/datasets/multiRAW/huawei_p30pro/test.txt',
        img_prefix='/lzh/datasets/multiRAW/huawei_p30pro/',
        img_subdir='camera_isp',
        meta_subdir='place_holder',
        ann_subdir='labels/detection',
        img_suffix='jpg',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2016, 1512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[0, 0, 0],
                        std=[255.0, 255.0, 255.0],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        rearrange_bbox_at_test=False),
    test=dict(
        type='MultiRAWDataset',
        ann_file='/lzh/datasets/multiRAW/huawei_p30pro/test.txt',
        img_prefix='/lzh/datasets/multiRAW/huawei_p30pro/',
        img_subdir='camera_isp',
        meta_subdir='place_holder',
        ann_subdir='labels/detection',
        img_suffix='jpg',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2016, 1512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[0, 0, 0],
                        std=[255.0, 255.0, 255.0],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        rearrange_bbox_at_test=False))
evaluation = dict(interval=1, metric='mAP')
model = dict(
    type='YOLOV3',
    backbone=dict(
        type='MobileNetV2RAW',
        in_ch=3,
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
work_dir = './work_dirs/yolov3_mobilenetv2_mstrain-2016_300e_huawei_p30pro'
gpu_ids = range(0, 4)
