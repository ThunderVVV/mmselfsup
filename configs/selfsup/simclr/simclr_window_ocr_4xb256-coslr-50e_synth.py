_base_ = [
    '../_base_/models/ocrsimclr_window.py',
    '../_base_/datasets/synth_simclr.py',
    '../_base_/schedules/adadelta_steplr-50e_synth.py',
    '../_base_/default_runtime.py',
]


# runtime settings
# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
checkpoint_config = dict(interval=10, max_keep_ckpts=3)
