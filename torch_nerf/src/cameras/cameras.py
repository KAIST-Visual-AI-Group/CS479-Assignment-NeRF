"""
Camera classes used inside renderer(s).
"""

from dataclasses import dataclass
from typing import Union
from jaxtyping import Float, Int, jaxtyped
from beartype import beartype as typechecker

import torch
from torch import Tensor

from torch_nerf.src.cameras.rays import RayBundle

@dataclass(init=False)
class Camera:
    """
    Camera class.
    """

    camera_to_world: Float[Tensor, "3 4"]
    """Camera-to-World matrix of shape [3, 4]."""
    f_x: Float[Tensor, "1"]
    """Focal length along the x-axis."""
    f_y: Float[Tensor, "1"]
    """Focal length along the y-axis."""
    c_x: Float[Tensor, "1"]
    """Principal point along the x-axis."""
    c_y: Float[Tensor, "1"]
    """Principal point along the y-axis."""
    near: Float[Tensor, "1"]
    """Near clipping plane."""
    far: Float[Tensor, "1"]
    """Far clipping plane."""
    image_width: Int[Tensor, "1"]
    """Image width."""
    image_height: Int[Tensor, "1"]
    """Image height."""
    device: Union[str, torch.device]
    """Device where camera information is stored."""

    @jaxtyped(typechecker=typechecker)
    def __init__(
        self,
        camera_to_world: Float[Tensor, "3 4"],
        f_x: Union[float, Float[Tensor, "1"]],
        f_y: Union[float, Float[Tensor, "1"]],
        c_x: Union[float, Float[Tensor, "1"]],
        c_y: Union[float, Float[Tensor, "1"]],
        near: Union[float, Float[Tensor, "1"]],
        far: Union[float, Float[Tensor, "1"]],
        image_width: Union[int, Int[Tensor, "1"]],
        image_height: Union[int, Int[Tensor, "1"]],
        device: Union[str, torch.device],
    ):
        """
        Initializes camera.
        """
        self.camera_to_world = camera_to_world
        self.f_x = f_x
        self.f_y = f_y
        self.c_x = c_x
        self.c_y = c_y
        self.near = near
        self.far = far
        self.image_width = image_width
        self.image_height = image_height
        self.device = device

        # cast numeric values to tensors
        for variable, value in vars(self).items():
            if isinstance(value, int):
                setattr(self, variable, torch.tensor(getattr(self, variable), dtype=torch.int))
            elif isinstance(value, float):
                setattr(self, variable, torch.tensor(getattr(self, variable), dtype=torch.float))

        # transfer tensors to device
        for variable, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, variable, getattr(self, variable).to(device))


    def __len__(self) -> int:
        """
        Returns the number of cameras.
        """
        return self.camera_to_world.shape[0]

    @jaxtyped(typechecker=typechecker)
    def generate_screen_coords(self) -> Int[Tensor, "num_pixel 2"]:
        """
        Generates screen coordinates corresponding to image pixels.
        
        The origin of the coordinate frame is located at the top left corner
        of an image, with the x-axis pointing to the right and the y-axis pointing
        downwards.
        """

        image_height = self.image_height.item()
        image_width = self.image_width.item()
        device = self.device

        i_indices = torch.arange(0, image_height, device=device)
        j_indices = torch.arange(0, image_width, device=device)
        i_grid, j_grid = torch.meshgrid(i_indices, j_indices, indexing="ij")

        coords = torch.stack([j_grid, i_grid], dim=-1)
        coords = coords.reshape(image_height * image_width, 2)

        return coords

    @jaxtyped(typechecker=typechecker)
    def generate_ray_directions(
        self,
        screen_coords: Int[Tensor, "num_pixel 2"],
    ) -> Float[Tensor, "num_pixel 3"]:
        """
        Computes ray directions for the current camera.
        The direction vectors are represented in the camera frame.
        """

        f_x = self.f_x.item()
        f_y = self.f_y.item()
        c_x = self.c_x.item()
        c_y = self.c_y.item()

        screen_xs = screen_coords[..., 0]
        screen_ys = screen_coords[..., 1]

        ray_xs = (screen_xs - c_x) / f_x
        ray_ys = -(screen_ys - c_y) / f_y
        ray_zs = -torch.ones_like(ray_xs)
        ray_directions = torch.stack([ray_xs, ray_ys, ray_zs], dim=-1)

        return ray_directions

    @jaxtyped(typechecker=typechecker)
    def generate_rays(self) -> RayBundle:
        """
        Generates rays for the current camera.
        """

        # retrieve near and far bounds
        near = self.near.item()
        far = self.far.item()

        # compute ray direction(s)
        screen_coords = self.generate_screen_coords()
        ray_directions = self.generate_ray_directions(screen_coords)
        ray_directions = torch.sum(
            ray_directions[..., None, :] * self.camera_to_world[:3, :3], dim=-1
        )

        # compute ray origin(s)
        ray_origins = self.camera_to_world[:3, 3].expand(ray_directions.shape)

        # create ray bundle
        ray_bundle = RayBundle(
            origins=ray_origins,
            directions=ray_directions,
            nears=torch.ones_like(ray_directions[..., 0]) * near,
            fars=torch.ones_like(ray_directions[..., 0]) * far,
        )

        return ray_bundle
