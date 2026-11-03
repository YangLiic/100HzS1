"""Preserve reference-sensitive operations in TensorRT FP16 deployment."""

PRECISION_RECIPE = 'fp16_sensitive_reference_ops_fp32_v1'


def protect_fp16_sensitive_layers(network, config, trt):
    """Keep EfficientViT float attention, normalization and softmax in FP32.

    Conv/Linear operators remain eligible for FP16. Shape/integer operators
    must retain their original types. This is a build-time policy only.
    """
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    sensitive = {trt.LayerType.MATRIX_MULTIPLY, trt.LayerType.ELEMENTWISE,
                 trt.LayerType.REDUCE, trt.LayerType.UNARY,
                 trt.LayerType.ACTIVATION, trt.LayerType.SOFTMAX}
    selected = []
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        name = layer.name.lower()
        attention = ('context_module/' in name and
                     not any(x in name for x in ['/qkv/', '/aggreg', '/proj/']))
        norm = 'norm' in name and layer.type in {
            trt.LayerType.ELEMENTWISE, trt.LayerType.REDUCE, trt.LayerType.UNARY}
        if layer.type not in sensitive or not (attention or norm or layer.type == trt.LayerType.SOFTMAX):
            continue
        outputs = [layer.get_output(j) for j in range(layer.num_outputs)]
        if not outputs or any(t is None or t.dtype not in (trt.float32, trt.float16) for t in outputs):
            continue
        layer.precision = trt.float32
        for j in range(layer.num_outputs):
            layer.set_output_type(j, trt.float32)
        selected.append(layer.name)
    return selected
