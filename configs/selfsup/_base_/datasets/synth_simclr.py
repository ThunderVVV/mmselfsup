# dataset settings
data_source = 'ImageList'
dataset_type = 'MultiViewDataset'
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_pipeline = [
    dict(type='Resize', size=[32, 100]),
    dict(type='MyTransform'),
    ]

# prefetch
prefetch = False
if not prefetch:
    train_pipeline.extend(
        [dict(type='ToTensor'),
         dict(type='Normalize', **img_norm_cfg)])

# dataset summary
data = dict(
    samples_per_gpu=256,  # total 256*4
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='data/synth',
            ann_file='data/synth.txt',
        ),
        # data_source=dict(
        #     type=data_source,
        #     data_prefix='/remote-home/jlzhang/OCR2022/mmselfsup/data/lmdb/synth',
        #     ann_file='/remote-home/jlzhang/OCR2022/mmselfsup/data/lmdb/synth',
        # ),
        num_views=[2],
        pipelines=[train_pipeline],
        prefetch=prefetch,
    ))
