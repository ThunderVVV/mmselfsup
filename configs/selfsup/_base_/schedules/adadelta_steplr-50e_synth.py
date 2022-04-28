# optimizer
optimizer = dict(type='Adadelta', lr=10, weight_decay=1e-4, rho=0.95)
optimizer_config = dict(grad_clip=dict(max_norm=5))  # grad_clip, coalesce, bucket_size_mb

# learning policy
lr_config = dict(policy='step', step=[30, 40])

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=50)
