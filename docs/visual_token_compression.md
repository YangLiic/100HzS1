# Visual compression and width variants

Current B0 development supports exactly four variants, documented in
[the four computation paths](beat_b0_four_visual_variants_20261102.md) and
[the configuration index](../configs/100hzs1_train/README.md#b0-visual-variants).

All variants pool to 2x2 after multiview fusion. Shared-stage4 variants compress
each view to its natural 2x3 grid before fusion, using the same frozen stage4
twice and one trainable shared 128->64 bridge. RMS variants retain the visual
Linear128->1024 and use a pre-RMS residual visual MLP to keep the final width1024.

Previous compact/prefusion-ResBlock/independent-MBConv recipes and experimental
switches have been removed. Git history retains historical implementation and
results; those historical paths are not supported current training recipes.
