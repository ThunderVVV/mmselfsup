# model settings
model = dict(
    type='SimCLR',
    backbone=dict(
        type='CustomizedBackbone',
        SelfSL_layer="CNNLSTM",
        instance_map="window"),
    neck=dict(
        type='NoneNeck',
        with_avg_pool=True),
    head=dict(type='ContrastiveHead', temperature=0.1))
