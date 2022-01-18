_base_ = ['../_base_/models/yolo_raw.py', '../_base_/datasets/iphone_xsmax_raw.py']
model = dict(bbox_head=dict(num_classes=1))
# dataset settings
dataset_type = 'MultiRAWCarOnlyDataset'
data = dict(
    train=dict(dataset=dict(type=dataset_type)),
    val=dict(type=dataset_type),
    test=dict(type=dataset_type))
