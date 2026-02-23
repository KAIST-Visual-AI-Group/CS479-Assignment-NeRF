"""A set of utility functions commonly used in training/testing scripts."""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

from omegaconf import DictConfig
import torch
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torch_nerf.src.network as network
import torch_nerf.src.scene as scene
import torch_nerf.src.renderer.integrators.quadrature_integrator as integrators
import torch_nerf.src.renderer.ray_samplers as ray_samplers
from torch_nerf.src.renderer.volume_renderer import VolumeRenderer
from torch_nerf.src.signal_encoder import PositionalEncoder
from torch_nerf.src.utils.data.blender_dataset import NeRFBlenderDataset
from torch_nerf.src.utils.data.llff_dataset import LLFFDataset


def init_torch(cfg: DictConfig):
    """
    Initializes PyTorch with the given configuration.
    """
    # set random seed for PyTorch
    torch.manual_seed(42)

    # restrict the number of threads used by the current process
    torch.set_num_threads(1)


def init_cuda(cfg: DictConfig):
    """
    Checks availability of CUDA devices in the system and set the default device.
    """
    if torch.cuda.is_available():
        device_id = cfg.cuda.device_id

        if device_id > torch.cuda.device_count() - 1:
            print(
                "Invalid device ID. "
                f"There are {torch.cuda.device_count()} devices but got index {device_id}."
            )
            device_id = 0
            cfg.cuda.device_id = device_id  # overwrite config
            print(f"Set device ID to {cfg.cuda.device_id} by default.")
        torch.cuda.set_device(cfg.cuda.device_id)
        print(f"CUDA device detected. Using device {torch.cuda.current_device()}.")
    else:
        print("CUDA is not supported on this system. Using CPU by default.")


def init_dataset_and_loader(cfg: DictConfig):
    """
    Initializes the dataset and loader.
    """
    if cfg.data.dataset_type == "nerf_synthetic":
        dataset = NeRFBlenderDataset(
            cfg.data.data_root,
            scene_name=cfg.data.scene_name,
            data_type=cfg.data.data_type,
            half_res=cfg.data.half_res,
            white_bg=cfg.data.white_bg,
        )
    elif cfg.data.dataset_type == "nerf_llff":
        dataset = LLFFDataset(
            cfg.data.data_root,
            scene_name=cfg.data.scene_name,
            factor=cfg.data.factor,
            recenter=cfg.data.recenter,
            bd_factor=cfg.data.bd_factor,
            spherify=cfg.data.spherify,
        )

        # update the near and far bounds
        if cfg.renderer.project_to_ndc:
            cfg.renderer.t_near = 0.0
            cfg.renderer.t_far = 1.0
            print(
                "Using NDC projection for LLFF scene. "
                f"Set (t_near, t_far) to ({cfg.renderer.t_near}, {cfg.renderer.t_far})."
            )
        else:
            cfg.renderer.t_near = float(torch.min(dataset.z_bounds) * 0.9)
            cfg.renderer.t_far = float(torch.max(dataset.z_bounds) * 1.0)
            print(
                "Proceeding without NDC projection. "
                f"Set (t_near, t_far) to ({cfg.renderer.t_near}, {cfg.renderer.t_far})."
            )
    elif cfg.data.dataset_type == "nerf_deepvoxels":
        raise NotImplementedError()
    else:
        raise ValueError("Unsupported dataset.")

    print("===========================================")
    print("Loaded dataset successfully.")
    print(f"Dataset type / Scene name: {cfg.data.dataset_type} / {cfg.data.scene_name}")
    print(f"Number of training data: {len(dataset)}")
    print(f"Image resolution: ({dataset.img_height}, {dataset.img_width})")
    print(f"Focal length(s): ({dataset.focal_length}, {dataset.focal_length})")
    print("===========================================")

    loader = data.DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=cfg.data.shuffle,
        num_workers=4,  # TODO: Adjust dynamically according to cfg.cuda.device_id
    )

    return dataset, loader


def init_renderer(cfg: DictConfig):
    """
    Initializes the renderer for rendering scene representations.
    """
    integrator = None
    sampler = None

    if cfg.renderer.integrator_type == "quadrature":
        integrator = integrators.QuadratureIntegrator()
    else:
        raise ValueError("Unsupported integrator type.")

    if cfg.renderer.sampler_type == "stratified":
        sampler = ray_samplers.StratifiedSampler()
    else:
        raise ValueError("Unsupported ray sampler type.")

    renderer = VolumeRenderer(integrator, sampler)

    return renderer


def init_tensorboard(tb_log_dir: str):
    """
    Initializes tensorboard writer.
    """
    if not os.path.exists(tb_log_dir):
        os.mkdir(tb_log_dir)
    writer = SummaryWriter(log_dir=tb_log_dir)
    return writer


def init_scene(cfg: DictConfig):
    """
    Initializes the scene representation to be trained / tested.
    """
    if cfg.signal_encoder.type == "pe":
        coord_enc = PositionalEncoder(
            cfg.network.pos_dim,
            cfg.signal_encoder.coord_encode_level,
            cfg.signal_encoder.include_input,
        )
        dir_enc = PositionalEncoder(
            cfg.network.view_dir_dim,
            cfg.signal_encoder.dir_encode_level,
            cfg.signal_encoder.include_input,
        )
    elif cfg.signal_encoder.type == "sh":
        coord_enc = SHEncoder(
            cfg.network.pos_dim,
            cfg.signal_encoder.degree,
        )
        dir_enc = SHEncoder(
            cfg.network.view_dir_dim,
            cfg.signal_encoder.degree,
        )
    else:
        raise NotImplementedError()
    encoders = {
        "coord_enc": coord_enc,
        "dir_enc": dir_enc,
    }
    if cfg.scene.type == "cube":
        if cfg.network.type == "nerf":
            default_network = network.NeRF(
                coord_enc.out_dim,
                dir_enc.out_dim,
            ).to(cfg.cuda.device_id)
        elif cfg.network.type == "instant_nerf":
            default_network = network.InstantNeRF(
                cfg.network.pos_dim,
                dir_enc.out_dim,
                cfg.network.num_level,
                cfg.network.log_max_entry_per_level,
                cfg.network.min_res,
                cfg.network.max_res,
                table_feat_dim=cfg.network.table_feat_dim,
            ).to(cfg.cuda.device_id)
            encoders.pop("coord_enc", None)
        else:
            raise NotImplementedError()

        default_scene = scene.PrimitiveCube(
            default_network,
            encoders,
        )

        fine_scene = None
        if cfg.renderer.num_samples_fine > 0:  # initialize fine scene
            if cfg.network.type == "nerf":
                fine_network = network.NeRF(
                    coord_enc.out_dim,
                    dir_enc.out_dim,
                ).to(cfg.cuda.device_id)
            elif cfg.network.type == "instant_nerf":
                fine_network = network.InstantNeRF(
                    cfg.network.pos_dim,
                    dir_enc.out_dim,
                    cfg.network.num_level,
                    cfg.network.log_max_entry_per_level,
                    cfg.network.min_res,
                    cfg.network.max_res,
                    table_feat_dim=cfg.network.table_feat_dim,
                ).to(cfg.cuda.device_id)
            else:
                raise NotImplementedError()
            fine_scene = scene.PrimitiveCube(
                fine_network,
                encoders,
            )
        return default_scene, fine_scene
    else:
        raise ValueError("Unsupported scene representation.")


def init_optimizer_and_scheduler(
    cfg: DictConfig,
    default_scene,
    fine_scene=None,
):
    """
    Initializes the optimizer and learning rate scheduler used for training.
    """
    optimizer = None
    scheduler = None

    # identify parameters to be optimized
    params = list(default_scene.radiance_field.parameters())
    if not fine_scene is None:
        params += list(fine_scene.radiance_field.parameters())

    # ==============================================================================
    # configure optimizer
    if cfg.train_params.optim.optim_type == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=cfg.train_params.optim.init_lr,
            eps=cfg.train_params.optim.eps,
        )
    else:
        raise NotImplementedError()

    # ==============================================================================
    # configure learning rate scheduler
    if cfg.train_params.optim.scheduler_type == "exp":
        # compute decay rate
        init_lr = cfg.train_params.optim.init_lr
        end_lr = cfg.train_params.optim.end_lr
        num_iter = cfg.train_params.optim.num_iter
        gamma = pow(end_lr / init_lr, 1 / num_iter)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma,
        )
    else:
        raise NotImplementedError()

    return optimizer, scheduler


def init_loss_func(cfg: DictConfig):
    """
    Initializes objective functions used to train neural radiance fields.
    """
    if cfg.objective.loss_type == "nerf_default":
        return torch.nn.MSELoss()
    else:
        raise ValueError("Unsupported loss configuration.")


def save_ckpt(
    ckpt_dir: str,
    epoch: int,
    default_scene: scene.scene,
    fine_scene: scene.scene,
    optimizer: torch.optim.Optimizer,
    scheduler,
):
    """
    Saves the checkpoint.
    """
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt_file = os.path.join(ckpt_dir, f"ckpt_{str(epoch).zfill(6)}.pth")

    ckpt = {
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
    }

    # save scheduler state
    if not scheduler is None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()

    # save scene(s)
    ckpt["scene_default"] = default_scene.radiance_field.state_dict()
    if not fine_scene is None:
        ckpt["scene_fine"] = fine_scene.radiance_field.state_dict()

    torch.save(
        ckpt,
        ckpt_file,
    )


def load_ckpt_from_file(
    ckpt_path: Path,
    default_scene: scene.scene,
    fine_scene: scene.scene = None,
    optimizer: torch.optim.Optimizer = None,
    scheduler: object = None,
):
    """
    Loads the checkpoint from a single file.
    """
    print(f"Loading checkpoint from: {str(ckpt_path)}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # load epoch
    epoch = ckpt.get("epoch", 0)

    # load scene(s)
    default_scene.radiance_field.load_state_dict(ckpt["scene_default"])
    default_scene.radiance_field.to(torch.cuda.current_device())
    if fine_scene is not None and "scene_fine" in ckpt:
        fine_scene.radiance_field.load_state_dict(ckpt["scene_fine"])
        fine_scene.radiance_field.to(torch.cuda.current_device())

    # load optimizer and scheduler states if provided
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    print("Checkpoint loaded.")
    return epoch


def load_ckpt(
    ckpt_dir: Path,
    default_scene: scene.scene,
    fine_scene: scene.scene,
    optimizer: torch.optim.Optimizer,
    scheduler: object = None,
):
    """
    Loads the latest checkpoint from a directory.
    """
    epoch = 0

    if ckpt_dir is None or not ckpt_dir.exists():
        print("Checkpoint directory not found.")
        return epoch

    ckpt_files = sorted(list(ckpt_dir.iterdir()))
    if len(ckpt_files) == 0:
        print("No checkpoint file found.")
        return epoch

    ckpt_file = ckpt_files[-1]
    print(f"Loading the latest checkpoint: {str(ckpt_file)}")

    return load_ckpt_from_file(
        ckpt_file,
        default_scene,
        fine_scene,
        optimizer,
        scheduler,
    )
