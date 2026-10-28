# VFA development conventions

- Make remote edits directly over SSH when working in this checkout; a Mac mirror is not required. Inspect current files and preserve existing changes.
- New training launchers use the correctly spelled train, with the universal entry point at scripts/train/train.sh. Do not introduce the historical thrain spelling.
- Default VFA training checkpoints belong under /DRoboticsAILab/users/licheng.yang/checkpoints/VFA. Keep checkpoint_root centralized in the shared VFA policy config and retain task/recipe subdirectories. Explicit user overrides take precedence.
- VFA changes normally belong in src/vfa/ and configs/vfa_*. Preserve the src/rtpolicy/ and configs/rtpolicy_* baseline unless the user requests changes there.
- LeRobot episode shared caching is managed by the VFA dataset. Training scripts should pass through the resolved dataset options, not launch a second cache service.
- Validate launcher changes with bash -n, dry runs and resolved configs before starting any GPU training.
- The offline dataset /DRoboticsAILab/vepfs_20260622/licheng.yang/datasets/robotwin2.0_lerobot_clean_joint14 contains BGR images. Set train_dataset.bgr_to_rgb=true for training and validation, including uncached baselines. Source color correction is independent of performance flags; online SAPIEN observations are RGB and must not be flipped for this reason.
- Keep bgr_to_rgb=false as the code and generic demo dataset default. Enable it by default only in RoboTwin dataset profiles such as robotwin2_10tasks.yaml and robotwin2_singletask.yaml; an explicit override remains available for already-RGB variants.
