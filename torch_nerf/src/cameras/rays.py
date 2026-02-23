"""
rays.py
"""

from dataclasses import dataclass
from beartype import beartype as typechecker

from jaxtyping import Float, jaxtyped
import torch


@dataclass(init=False)
class RayBundle:
    """
    Ray bundle class.
    """

    origins: Float[torch.Tensor, "*batch_size 3"]
    """Ray origins in the world coordinate."""
    directions: Float[torch.Tensor, "*batch_size 3"]
    """Ray directions in the world coordinate."""
    nears: Float[torch.Tensor, "*batch_size 1"]
    """Near clipping plane."""
    fars: Float[torch.Tensor, "*batch_size 1"]
    """Far clipping plane."""

    def __init__(
        self,
        origins: Float[torch.Tensor, "*batch_size 3"],
        directions: Float[torch.Tensor, "*batch_size 3"],
        nears: Float[torch.Tensor, "*batch_size 1"],
        fars: Float[torch.Tensor, "*batch_size 1"],
    ) -> None:
        """
        Initializes RayBundle.
        """
        self.origins = origins
        self.directions = directions
        self.nears = nears
        self.fars = fars

    def __len__(self) -> int:
        """
        Returns the number of rays in the bundle.
        """
        return self.origins.shape[0]

@dataclass(init=False)
class RaySamples:
    """
    Ray sample class.
    """

    ray_bundle: RayBundle
    """Ray bundle. Contains ray origin, direction, near and far bounds."""
    t_samples: Float[torch.Tensor, "num_ray num_sample"]
    """Distance values sampled along rays."""

    def __init__(
        self,
        ray_bundle: RayBundle,
        t_samples: Float[torch.Tensor, "num_ray num_sample"],
    ) -> None:
        """
        Initializes RaySample.
        """
        self.ray_bundle = ray_bundle
        self.t_samples = t_samples

    @jaxtyped(typechecker=typechecker)
    def compute_sample_coordinates(self) -> Float[torch.Tensor, "num_ray num_sample 3"]:
        """
        Computes coordinates of points sampled along rays in the ray bundle.

        Returns:
            coords: Coordinates of points sampled along rays in the ray bundle.
        """
        ray_origins: Float[torch.Tensor, "num_ray 3"] = self.ray_bundle.origins
        ray_directions: Float[torch.Tensor, "num_ray 3"] = self.ray_bundle.directions
        t_samples: Float[torch.Tensor, "num_ray num_sample"] = self.t_samples

        # ============================
        # Task 1-1. Implement Ray Sampling
        # DO NOT change the code outside this part.
        
        # ============================

        return sample_coords

    @jaxtyped(typechecker=typechecker)
    def compute_deltas(self, right_end: float=1e8) -> Float[torch.Tensor, "num_ray num_sample"]:
        """
        Compute differences between adjacent t's required to approximate integrals.

        Args:
            right_end: The value to be appended to the right end
                when computing 1st order difference.

        Returns:
            deltas: Differences between adjacent t's.
                When evaluating the delta for the farthest sample on a ray,
                use the value of the argument 'right_end'.
        """
        t_samples: Float[torch.Tensor, "num_ray num_sample"] = self.t_samples
        num_ray = t_samples.shape[0]
        device = t_samples.device

        deltas = torch.diff(
            t_samples,
            n=1,
            dim=-1,
            append=right_end * torch.ones((num_ray, 1), device=device),
        )

        return deltas
