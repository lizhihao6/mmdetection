# dataset settings
_base_ = ['rggb_to_ryyb.py']
data_root = '/lzh/datasets/multiRAW/oneplus_5t/'
suffix = 'dng'
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
