# dataset settings
_base_ = ['raw.py']
data_root = '/lzh/datasets/multiRAW/asi_294mcpro/'
suffix = 'raw.TIF'
data = dict(
    train=dict(
        dataset=dict(
            ann_file=data_root + 'train.txt',
            img_prefix=data_root,
            img_suffix=suffix)),
    val=dict(
        ann_file=data_root + 'test.txt',
        img_prefix=data_root,
        img_suffix=suffix),
    test=dict(
        ann_file=data_root + 'test.txt',
        img_prefix=data_root,
        img_suffix=suffix))
