"""Compilation policy, applied AFTER Accelerate/DDP preparation.

Raw parameter storage and checkpoint names stay unchanged. PyTorch3D FPS/KNN
remain eager graph breaks. Regional UNet compilation keeps ragged L0 eager by
default and compiles the coarse levels. Adaptive allocation can change coarse
shapes too; static compilation specializes on each encountered shape.
"""
import torch


def prepare_compilation(model, config, *, is_main_process=True):
    if not config.get("compile", False):
        return model
    import torch._dynamo as dynamo

    dynamic = config.get("compile_dynamic")
    if dynamic is False:
        dynamo.config.automatic_dynamic_shapes = False
        # Shared code objects need room for shape, width, train/eval and dtype variants.
        for name, minimum in (("cache_size_limit", 256), ("recompile_limit", 256),
                              ("accumulated_recompile_limit", 1024)):
            if hasattr(dynamo.config, name):
                setattr(dynamo.config, name, max(minimum, getattr(dynamo.config, name)))
    # DDPOptimizer cannot trace the activation-checkpoint higher-order operations.
    dynamo.config.optimize_ddp = False
    if not config.get("compile_regional", False):
        if is_main_process:
            print(f"torch.compile: whole model, dynamic={dynamic}; cold start can be slow")
        return torch.compile(model, dynamic=dynamic)

    from models.unet_std import Block, Pool, UNet, Unpool
    from models.joint_block import JointBlock
    from models.tube_joint import TubeJointCore
    # model.unet_core=tube_joint swaps the Blocks for JointBlocks but keeps the ladder, the seams and the
    # checkpointing knobs, so it compiles by exactly the same regions and the same two skip rules.
    CORES = (UNet, TubeJointCore)
    units = list(model.modules())
    networks = [unit for unit in units if isinstance(unit, CORES)]
    if not networks:
        raise ValueError("Regional compilation requires a UNet or a tube_joint core; "
                         "use whole-model compilation for other architectures")
    mode = config.get("compile_mode")
    skip_checkpoint, skip_l0 = set(), set()
    for network in networks:
        # which levels are checkpointed: the joint core resolves ckpt_from_level (and any explicit
        # unet_tube_ckpt_levels override) into a level set at build time.
        ckpt = (sorted(network.ckpt_levels) if isinstance(network, TubeJointCore)
                else ([l for l in range(int(network.cfg.ckpt_from_level), len(network.enc))]
                      if int(network.cfg.ckpt_from_level) >= 0 else []))
        if not network.cfg.ckpt_ingraph and ckpt:
            # External checkpoint recomputation must not mix compiled/eager Block metadata.
            for index in ckpt:
                skip_checkpoint.update(id(block) for block in network.enc[index])
                if index < len(network.dec):
                    skip_checkpoint.update(id(block) for block in network.dec[index])
        if network.cfg.stratified and not config.get("compile_l0", False):
            skip_l0.update(id(unit) for unit in (
                *network.enc[0], *network.dec[0], network.pools[0], network.unpools[0],
                network.stem, network.head))
    count = 0
    for unit in units:
        if isinstance(unit, CORES):
            candidates = (unit.stem, unit.head, unit.t_mlp)
        elif isinstance(unit, (Block, JointBlock, Pool, Unpool)):
            candidates = (unit,)
        else:
            continue
        for candidate in candidates:
            if id(candidate) not in skip_checkpoint | skip_l0:
                candidate.compile(dynamic=dynamic, mode=mode)
                count += 1
    if is_main_process:
        print(f"torch.compile: regional, dynamic={dynamic}, mode={mode}; {count} regions; "
              f"{len(skip_checkpoint)} externally checkpointed blocks and {len(skip_l0)} L0 units eager")
    return model
