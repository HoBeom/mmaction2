_base_ = [
    '../../_base_/models/i3d_r50.py', '../../_base_/schedules/sgd_100e.py',
    '../../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(in_channels=2),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[128, 128],
        std=[128, 128],
        to_rgb=False,
        format_shape='NCTHW'))

# dataset settings
dataset_type = 'VideoDataset'
data_root = 'data/kinetics400/flow_videos_train_lmdb'
data_root_val = 'data/kinetics400/flow_videos_val_lmdb'
data_root_test = 'data/kinetics400/flow_videos_test_lmdb'
ann_file_train = 'data/kinetics400/kinetics400_train_list_flow_videos_lmdb.txt'
ann_file_val = 'data/kinetics400/kinetics400_val_list_flow_videos_lmdb.txt'
ann_file_test = 'data/kinetics400/kinetics400_test_list_flow_videos_lmdb.txt'

# file_client_args = dict(io_backend='disk', num_threads=1)
train_pipeline = [
    dict(
        type='VideoLmdbInit',
        io_backend='lmdb',
        db_path=data_root,
        num_threads=4,
        map_size=8.4e+10,
        max_readers=64,
        max_spare_txns=64),
    dict(type='UniformSampleFrames', clip_len=32, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.8),
        random_crop=False,
        max_wh_scale_gap=0),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW_Flow'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(
        type='VideoLmdbInit',
        io_backend='lmdb',
        db_path=data_root_val,
        num_threads=4,
        map_size=7e+9,
        max_readers=64,
        max_spare_txns=64),
    dict(type='UniformSampleFrames', clip_len=32, num_clips=1, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW_Flow'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(
        type='VideoLmdbInit',
        io_backend='lmdb',
        db_path=data_root_test,
        num_threads=4,
        map_size=1.4e+10,
        max_readers=64,
        max_spare_txns=64),
    dict(
        type='UniformSampleFrames', clip_len=32, num_clips=10, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='FormatShape', input_format='NCTHW_Flow'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=128,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        modality='Flow',
        ann_file=ann_file_train,
        data_prefix=dict(video=''),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=128,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        modality='Flow',
        ann_file=ann_file_val,
        data_prefix=dict(video=''),
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        modality='Flow',
        ann_file=ann_file_test,
        data_prefix=dict(video=''),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

default_hooks = dict(checkpoint=dict(interval=5, max_keep_ckpts=5))
# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=128)
