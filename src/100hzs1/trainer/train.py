import copy
import logging
import math
import os
from pathlib import Path
import hydra
from omegaconf import OmegaConf

import diffusers
import numpy as np
import torch
from torch.utils.data import RandomSampler, BatchSampler
import transformers
from transformers import AutoConfig, AutoModel
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from tqdm.auto import tqdm
from safetensors.torch import load_model

from ..common.sampler import ResumableEpochSampler
from ..common.ema_model import EMAModel
from ..common.checkpoint import validate_class_path


if is_wandb_available():
    import wandb

@torch.no_grad()
def log_sample_res(policy_model, args, dataloader, logger):
    logger.info(f"Running sampling for {args.num_val_batches} batches (bs={args.val_batch_size})...")

    policy_model.eval()
    
    loss_for_log = {}
    val_losses = {}
    for step, batch in enumerate(dataloader):
        if step >= args.num_val_batches:
            break
        loss = policy_model(**batch)
        if isinstance(loss, dict):
            for k, v in loss.items():
                if k not in val_losses:
                    val_losses[k] = []
                val_losses[k].append(v.item())
        else:
            if 'loss' not in val_losses:
                val_losses['loss'] = []
            val_losses['loss'].append(loss.item())
        
    for k, v_list in val_losses.items():
        loss_for_log[k] = torch.mean(torch.tensor(v_list)).item()

    policy_model.train()
    torch.cuda.empty_cache()

    return dict(loss_for_log)

def save_policy_custom(polciy, save_path):
    import inspect
    import shutil
    PolicyClass = polciy.__class__
    PolicyConfigClass = polciy.config_class
    code_path = inspect.getfile(PolicyClass)
    code_name = os.path.basename(code_path)
    if hasattr(polciy.config, 'obs_encoder') and 'emvis_config' in polciy.config.obs_encoder:
        emvis_config = polciy.config.obs_encoder['emvis_config']
        emvis_config['load_backbone_pretrain'] = False
        # emvis_config['load_vggt_heads'] = False
        emvis_config['visualize'] = False
    # Checkpoints load through their installed-package pkg_map.
    shutil.copy(code_path, os.path.join(save_path, code_name))

    state_dict = polciy.state_dict()
    filtered_state_dict = dict(state_dict)
    polciy.save_pretrained(save_path, state_dict=filtered_state_dict, max_shard_size="10GB")

def resume_policy(ckp_path, ema_policy_model):
    load_model(ema_policy_model, os.path.join(ckp_path, "model.safetensors"), strict=False)
    return ema_policy_model

def build_optimizer_params(policy_model, args, logger):
    optimizer_param_groups = getattr(args, "optimizer_param_groups", None)
    if not optimizer_param_groups:
        return policy_model.parameters()

    param_groups_config = OmegaConf.to_container(optimizer_param_groups, resolve=True)
    named_params = list(policy_model.named_parameters())
    grouped_param_ids = set()
    param_groups = []

    for group_config in param_groups_config:
        group_name = group_config["name"]
        prefixes = group_config.get("name_prefixes") or [group_config["name_prefix"]]
        group = {
            key: value
            for key, value in group_config.items()
            if key not in {"name", "name_prefix", "name_prefixes"}
        }
        group_params = [
            param
            for name, param in named_params
            if param.requires_grad and any(name.startswith(prefix) for prefix in prefixes)
        ]
        if not group_params:
            logger.warning(f"Optimizer param group `{group_name}` matched no trainable parameters.")
            continue
        group["params"] = group_params
        param_groups.append(group)
        grouped_param_ids.update(id(param) for param in group_params)
        logger.info(
            f"Optimizer param group `{group_name}`: "
            f"{sum(param.numel() for param in group_params):,} parameters, lr={group.get('lr', args.optimizer.lr)}"
        )

    default_params = [
        param
        for _, param in named_params
        if param.requires_grad and id(param) not in grouped_param_ids
    ]
    if default_params:
        param_groups.insert(0, {"params": default_params})
        logger.info(
            f"Optimizer default param group: "
            f"{sum(param.numel() for param in default_params):,} parameters, lr={args.optimizer.lr}"
        )
    return param_groups

def apply_dataset_normalizer(args, train_dataset, policy_model, ema_policy_model, logger):
    normalizer_source = getattr(args, "normalizer_source", "dataset")

    if not hasattr(train_dataset, "get_normalizer"):
        logger.info("Train dataset has no get_normalizer; skipping normalizer setup.")
        return normalizer_source

    if normalizer_source == "dataset":
        normalizer = train_dataset.get_normalizer()
        policy_model.set_normalizer(normalizer)
        ema_policy_model.set_normalizer(normalizer)
        logger.info("Loaded normalizer from train dataset.")
    elif normalizer_source == "checkpoint":
        if not (hasattr(args, "checkpoint_dir") and args.checkpoint_dir):
            raise ValueError("normalizer_source=\"checkpoint\" requires checkpoint_dir.")
        logger.info("Keeping normalizer loaded from checkpoint.")
    elif normalizer_source == "none":
        logger.info("Skipping normalizer override.")
    else:
        raise ValueError(f"Unknown normalizer_source: {normalizer_source}")

    return normalizer_source



def _cfg_get(cfg, key, default=None):
    return getattr(cfg, key) if cfg is not None and hasattr(cfg, key) else default


def prepare_optimizer_config(args, logger):
    optimizer_config = OmegaConf.to_container(args.optimizer, resolve=True)
    scale_lr = optimizer_config.pop("scale_lr", False)
    lr_scale_rule = optimizer_config.pop("lr_scale_rule", "linear")
    base_lr_batch_size = optimizer_config.pop("base_lr_batch_size", None)

    if not scale_lr:
        return OmegaConf.create(optimizer_config)

    if "lr" not in optimizer_config:
        raise ValueError("optimizer.scale_lr requires optimizer.lr")

    if base_lr_batch_size is None or base_lr_batch_size <= 0:
        raise ValueError("optimizer.scale_lr requires optimizer.base_lr_batch_size > 0")

    current_lr_batch_size = args.train_batch_size * args.gradient_accumulation_steps
    if current_lr_batch_size <= 0:
        raise ValueError("optimizer.scale_lr requires positive train_batch_size and gradient_accumulation_steps")

    batch_scale = current_lr_batch_size / base_lr_batch_size
    if lr_scale_rule == "linear":
        lr_scale = batch_scale
    elif lr_scale_rule == "sqrt":
        lr_scale = math.sqrt(batch_scale)
    else:
        raise ValueError(f"Unknown optimizer.lr_scale_rule: {lr_scale_rule}")

    base_lr = optimizer_config["lr"]
    optimizer_config["lr"] = base_lr * lr_scale
    logger.info(
        "Scaled optimizer lr: "
        f"rule={lr_scale_rule}, base_lr={base_lr}, scaled_lr={optimizer_config['lr']}, "
        f"current_lr_batch_size={current_lr_batch_size}, base_lr_batch_size={base_lr_batch_size}"
    )

    return OmegaConf.create(optimizer_config)


def dataset_loader_options(dataset, default_num_workers, prefetch_factor=None):
    """Use configured workers; CUDA decoding and image RAM cache require zero."""
    cuda_decode = str(getattr(dataset, "decode_device", "cpu")).startswith("cuda")
    main_process_only = cuda_decode or bool(getattr(dataset, "image_cache_size_bytes", 0))
    num_workers = 0 if main_process_only else int(default_num_workers)
    pin_memory = False if cuda_decode else getattr(dataset, "pin_memory", True)
    options = {
        "num_workers": int(num_workers),
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    if prefetch_factor is not None:
        if int(prefetch_factor) != prefetch_factor or prefetch_factor < 1:
            raise ValueError("dataloader_prefetch_factor must be a positive integer")
        if num_workers > 0:
            options["prefetch_factor"] = int(prefetch_factor)
    return options


def build_validation_dataloader(args, val_dataset, accelerator=None):
    if val_dataset is None:
        return None
    num_processes = accelerator.num_processes if accelerator is not None else 1
    num_workers = args.dataloader_num_workers // num_processes
    return torch.utils.data.DataLoader(
        val_dataset, batch_size=args.val_batch_size, shuffle=True,
        **dataset_loader_options(val_dataset, num_workers, getattr(args, "dataloader_prefetch_factor", None)),
    )


def training_batch_iterator(loader, device, cuda_prefetch=False):
    iterator = iter(loader)
    if cuda_prefetch:
        from ..common.cuda_prefetch import CudaPrefetchIterator

        return CudaPrefetchIterator(iterator, device)
    return iterator


def build_train_dataloader(args, train_dataset, accelerator, num_train_samples):
    per_device_batch_size = args.train_batch_size // accelerator.num_processes
    resume_offset_batch_size = per_device_batch_size
    num_workers = args.dataloader_num_workers // accelerator.num_processes
    sampler_type = _cfg_get(_cfg_get(args, "train_sampler"), "type", "random_replacement")

    if sampler_type == "random_replacement":
        sampler = RandomSampler(
            train_dataset,
            replacement=True,
            num_samples=num_train_samples,
        )
    elif sampler_type == "resumable_epoch":
        sampler = ResumableEpochSampler(
            train_dataset,
            seed=args.seed or 0,
            batch_size=resume_offset_batch_size,
            num_processes=accelerator.num_processes,
        )
    else:
        raise ValueError(f"Unknown train sampler type: {sampler_type}")

    batch_sampler = BatchSampler(
        sampler,
        batch_size=per_device_batch_size,
        drop_last=True,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        **dataset_loader_options(train_dataset, num_workers, getattr(args, "dataloader_prefetch_factor", None)),
    )
    return train_dataloader, sampler, per_device_batch_size


def sampler_supports_epoch_resume(sampler):
    return all(
        hasattr(sampler, name)
        for name in (
            "set_epoch",
            "set_epoch_offset",
            "set_resume_batch_offset",
            "clear_resume_batch_offset",
        )
    )


def configure_resumable_sampler_for_resume(
    sampler,
    global_step,
    gradient_accumulation_steps,
    steps_per_epoch,
    supports_epoch_resume=None,
):
    if supports_epoch_resume is None:
        supports_epoch_resume = sampler_supports_epoch_resume(sampler)

    loader_epoch = 0
    batch_in_epoch = 0
    if supports_epoch_resume and global_step > 0:
        consumed_batches = global_step * gradient_accumulation_steps
        resume_epoch = consumed_batches // steps_per_epoch
        batch_in_epoch = consumed_batches % steps_per_epoch

        sampler.set_epoch(loader_epoch)
        sampler.set_epoch_offset(resume_epoch)
        sampler.set_resume_batch_offset(batch_in_epoch)

    return loader_epoch, batch_in_epoch

def train(args, logger):
    cuda_prefetch = bool(getattr(args, "dataloader_cuda_prefetch", False))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=Path(args.output_dir, args.logging_dir),
        project_config=ProjectConfiguration(total_limit=args.checkpoints_total_limit),
        dataloader_config=DataLoaderConfiguration(
            non_blocking=bool(getattr(args, "dataloader_non_blocking", False)),
        ),
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Log the accelerator state information on all processes for debugging
    # This includes information about distributed training setup, device allocation,
    # mixed precision settings, and other accelerator configuration details
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Policy Model creation
    model_args = OmegaConf.to_container(args.model)
    pkg_config = validate_class_path(model_args.pop('target_config'))
    pkg_policy = validate_class_path(model_args.pop('target_policy'))
    ConfigClass = hydra.utils.get_class(pkg_config)
    PolicyClass = hydra.utils.get_class(pkg_policy)
    model_args['pkg_map'] = {
        "AutoConfig": pkg_config,
        "AutoModel": pkg_policy
    }
    config = ConfigClass.from_dict(model_args)
    policy_model = PolicyClass(config)
    if hasattr(args, "checkpoint_dir") and args.checkpoint_dir:
        load_model(policy_model, os.path.join(args.checkpoint_dir, "model.safetensors"), strict=False)
    # policy_model = load_policy("checkpoints/dp_baseline/test", use_ckp_code = True)
    policy_model.to(accelerator.device)

    ema_policy_model = copy.deepcopy(policy_model)
    ema_model = EMAModel(
        ema_policy_model,
        update_after_step=args.ema.update_after_step,
        inv_gamma=args.ema.inv_gamma,
        power=args.ema.power,
        min_value=args.ema.min_value,
        max_value=args.ema.max_value,
    )

    from ..common.vision_compile import compile_frozen_vision_backbone

    vision_compile = compile_frozen_vision_backbone(
        policy_model, enabled=getattr(args, "vision_backbone_compile", False),
    )
    if vision_compile is not None:
        logger.info("Frozen vision backbone compilation enabled: %s", vision_compile)

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    # which ensure saving model in huggingface format (config.json + pytorch_model.bin)
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                model_to_save = model.module if hasattr(model, "module") else model  # type: ignore
                if isinstance(model_to_save, type(accelerator.unwrap_model(policy_model))):
                    save_policy_custom(model_to_save, output_dir)

    accelerator.register_save_state_pre_hook(save_model_hook)

    if args.gradient_checkpointing:
        # TODO: 
        raise NotImplementedError("Gradient checkpointing is not yet implemented.")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Optimizer creation
    optimizer_config = prepare_optimizer_config(args, logger)
    params_to_optimize = build_optimizer_params(policy_model, args, logger)
    if isinstance(params_to_optimize, list):
        # param groups already carry per-group lr overrides
        optimizer_config = OmegaConf.to_container(optimizer_config, resolve=True)
        optimizer_cls = hydra.utils.get_class(optimizer_config.pop("_target_"))
        optimizer = optimizer_cls(params=params_to_optimize, **optimizer_config)
    else:
        optimizer = hydra.utils.instantiate(optimizer_config, params=params_to_optimize)

    # Dataset and DataLoaders creation
    train_dataset = hydra.utils.instantiate(args.train_dataset)
    apply_dataset_normalizer(args, train_dataset, policy_model, ema_policy_model, logger)
    if (
        hasattr(train_dataset, "get_static_joint_info")
        and getattr(policy_model, "state_key", None) in {"state", "delta_state"}
        and hasattr(policy_model, "set_static_joint_mask")
    ):
        static_joint_info = train_dataset.get_static_joint_info()
        if static_joint_info is not None:
            static_joint_mask = static_joint_info["mask"]
            policy_model.set_static_joint_mask(static_joint_mask)
            ema_policy_model.set_static_joint_mask(static_joint_mask)
            logger.info(
                "Static joint mask configured: dims=%s, num_train_episodes=%d, "
                "ratio_threshold=%.6g, support_counts=%s",
                np.flatnonzero(static_joint_mask).tolist(),
                static_joint_info["num_episodes"],
                static_joint_info["ratio_threshold"],
                static_joint_info["support_counts"].tolist(),
            )
    val_dataset = train_dataset.get_validation_dataset()
    max_train_steps = getattr(args, "max_train_steps", None)
    if max_train_steps is not None:
        max_iters = int(max_train_steps)
        num_train_samples = max_iters * args.gradient_accumulation_steps * args.train_batch_size
    else:
        num_train_samples = len(train_dataset) * args.num_train_epochs
        max_iters = num_train_samples // args.train_batch_size

    train_dataloader, train_sampler, per_device_batch_size = build_train_dataloader(
        args,
        train_dataset,
        accelerator,
        num_train_samples,
    )
    val_dataloader = build_validation_dataloader(args, val_dataset, accelerator)

    lr_config = args.lr_scheduler
    num_lr_update_steps = max_iters * accelerator.num_processes
    lr_scheduler = get_scheduler(
        lr_config.name,
        optimizer=optimizer,
        num_warmup_steps=int(lr_config.warmup_ratio * num_lr_update_steps)
            if hasattr(lr_config, "warmup_ratio") else 0,
        num_training_steps=num_lr_update_steps,
        num_cycles=lr_config.num_cycles 
            if hasattr(lr_config, "num_cycles") else 1,
        power=lr_config.power 
            if hasattr(lr_config, "power") else 1.0,
    )

    # Prepare everything with `accelerator`.
    if cuda_prefetch and (
        accelerator.device.type != "cuda"
        or str(getattr(train_dataset, "decode_device", "cpu")).startswith("cuda")
        or not train_dataloader.pin_memory
        or args.gradient_accumulation_steps != 1
    ):
        raise ValueError("dataloader_cuda_prefetch requires CUDA training, a pinned CPU-decoded DataLoader and gradient_accumulation_steps=1")
    if val_dataloader is not None:
        policy_model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
            policy_model, optimizer, train_dataloader, val_dataloader, lr_scheduler,
            device_placement=[True, True, False, True, True] if cuda_prefetch else None,
        )
    else:
        policy_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            policy_model, optimizer, train_dataloader, lr_scheduler,
            device_placement=[True, True, False, True] if cuda_prefetch else None,
        )
    ema_policy_model.to(accelerator.device, dtype=weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        # Convert omegaconf.DictConfig to primitive types so it doesn't get dumped as truncated JSON
        config_dict = OmegaConf.to_container(args, resolve=True)
        accelerator.init_trackers("100HzS1", config=config_dict)

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {getattr(args, 'num_train_epochs', None)}")
    logger.info(f"  Max train steps = {max_train_steps}")
    logger.info(f"  Total sampled examples = {num_train_samples}")
    logger.info(f"  Batch size per device = {per_device_batch_size}")
    logger.info(f"  Num processes = {accelerator.num_processes}")
    logger.info(f"  Total batch size = {args.train_batch_size}")
    logger.info(f"  Num iter per device = {len(train_dataloader)}")
    logger.info(f"  Num optimization steps = {max_iters}")
    global_step = 0
    supports_epoch_resume = sampler_supports_epoch_resume(train_sampler)
    loader_epoch = 0
    batch_in_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            try:
                # torch>=2.6 defaults weights_only=True and accelerate's load() sets it
                # explicitly, breaking on omegaconf objects in saved states.
                # Force weights_only=False during resume load.
                _real_load = torch.load
                def _load_wo_false(*a, **k):
                    k["weights_only"] = False
                    return _real_load(*a, **k)
                torch.load = _load_wo_false
                try:
                    accelerator.load_state(os.path.join(args.output_dir, path), strict=False)   # TODO: drop strict=False
                finally:
                    torch.load = _real_load
            except:
                # load deepspeed's state_dict
                logger.info("Resuming training state failed. Attempting to only load from model checkpoint.")
                checkpoint = torch.load(os.path.join(args.output_dir, path, "pytorch_model", "mp_rank_00_model_states.pt"))
                policy_model.module.load_state_dict(checkpoint["module"])
            
            checkpoint_path = os.path.join(args.output_dir, path)
            global_step = int(path.split("-")[1])
            from ..common.training_resume import restore_normalizer_devices, restore_training_ema

            resume_details = restore_training_ema(
                ema_policy_model, ema_model, optimizer, checkpoint_path, global_step,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
            )
            logger.info("Restored training resume state: %s", resume_details)

            restore_normalizer_devices(
                (accelerator.unwrap_model(policy_model), ema_policy_model), accelerator.device,
            )

            loader_epoch, batch_in_epoch = configure_resumable_sampler_for_resume(
                train_sampler,
                global_step=global_step,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                steps_per_epoch=len(train_dataloader),
                supports_epoch_resume=supports_epoch_resume,
            )
            if supports_epoch_resume and hasattr(train_dataloader, "set_epoch"):
                train_dataloader.set_epoch(loader_epoch)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(0, max_iters), disable=not accelerator.is_local_main_process, initial=global_step)
    progress_bar.set_description("Steps")

    policy_model.train()

    # Forward and backward...
    data_iter = training_batch_iterator(train_dataloader, accelerator.device, cuda_prefetch)
    while global_step < max_iters:
        try:
            batch = next(data_iter)
            if supports_epoch_resume:
                batch_in_epoch += 1
        except StopIteration:
            if not supports_epoch_resume:
                break

            loader_epoch += 1
            batch_in_epoch = 0
            train_sampler.set_epoch(loader_epoch)
            train_sampler.clear_resume_batch_offset()
            if hasattr(train_dataloader, "set_epoch"):
                train_dataloader.set_epoch(loader_epoch)
            data_iter = training_batch_iterator(train_dataloader, accelerator.device, cuda_prefetch)
            continue

        with accelerator.accumulate(policy_model):
            loss = policy_model(**batch)
            if isinstance(loss, dict):
                loss_for_log = loss
                loss = loss.pop("loss")
            else:
                loss_for_log = {"loss": loss.detach().item()}

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                params_to_clip = policy_model.parameters()
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=args.set_grads_to_none)


        ema_model.step(accelerator.unwrap_model(policy_model))

        # Checks if the accelerator has performed an optimization step behind the scenes
        logs = {}
        if accelerator.sync_gradients:
            progress_bar.update(1)

            global_step += 1

            if args.checkpointing_period > 0 and global_step % args.checkpointing_period == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path, False)
                ema_save_path = os.path.join(save_path, f"ema")
                accelerator.save_model(ema_policy_model, ema_save_path)
                logger.info(f"Saved state to {save_path}")

            if val_dataloader is not None and args.val_period > 0 and global_step % args.val_period == 0:
                logger.info(f"Sampling at step {global_step}")
                sample_loss_for_log = log_sample_res(
                    policy_model,    # We do not use EMA currently
                    args,
                    val_dataloader,
                    logger,
                )
                # logger.info(sample_loss_for_log)
                # accelerator.log(sample_loss_for_log, step=global_step)
                loss_for_log.update({f"val {k}": v for k, v in sample_loss_for_log.items()})
        
        logs.update({"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]})
        progress_bar.set_postfix(**logs)
        logs.update(loss_for_log)
        # logger.info(logs)
        accelerator.log(logs, step=global_step)

        if global_step >= max_iters:
            break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # accelerator.unwrap_model(policy_model).save_pretrained(args.output_dir)
        model_to_save = accelerator.unwrap_model(policy_model)
        save_policy_custom(model_to_save, args.output_dir)
        # TODO change
        # model_to_save.save_pretrained(args.output_dir, max_shard_size="10GB")

        ema_save_path = os.path.join(args.output_dir, f"ema")
        accelerator.save_model(ema_policy_model, ema_save_path)
        
        logger.info(f"Saved Model to {args.output_dir}")

    accelerator.end_training()
