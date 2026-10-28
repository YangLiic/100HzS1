"""Training-only compilation of the frozen visual backbone."""


def compile_frozen_vision_backbone(policy_model, enabled=False):
    """Compile in place after copying EMA, before distributed wrapping.

    Kernel compilation is runtime state. Saved policies retain ordinary state
    keys and can be loaded for eager inference without compilation.
    """
    if not enabled:
        return None
    encoder = getattr(getattr(policy_model, "obs_encoder", None), "scene_encoder", None)
    backbone = getattr(encoder, "backbone", None)
    if backbone is None:
        raise ValueError("vision_backbone_compile requires a visual backbone")
    if any(parameter.requires_grad for parameter in backbone.parameters()):
        raise ValueError("vision_backbone_compile requires a frozen visual backbone")
    # Module.compile keeps checkpoint keys, unlike assigning the OptimizedModule
    # returned by torch.compile(backbone). Leave precision and layout unchanged.
    backbone.compile(backend="inductor", fullgraph=False, dynamic=False)
    return {"backend": "inductor", "fullgraph": False, "dynamic": False}
