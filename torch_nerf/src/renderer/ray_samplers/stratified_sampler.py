"""
Stratified sampler implementation.
"""

from typing import Optional, Union
from beartype import beartype as typechecker

from jaxtyping import Float, Shaped, jaxtyped
import torch

from torch_nerf.src.renderer.ray_samplers.sampler_base import RaySamplerBase
from torch_nerf.src.cameras.rays import RayBundle, RaySamples
from torch_nerf.src.renderer.ray_samplers.utils import sample_pdf


class StratifiedSampler(RaySamplerBase):
    """
    Stratified sampler that samples points along rays.
    """

    @jaxtyped(typechecker=typechecker)
    def sample_along_rays(
        self,
        ray_bundle: RayBundle,
        num_sample: int,
        importance_weights: Optional[Float[torch.Tensor, "num_ray num_sample"]] = None,
        importance_t_samples: Optional[Float[torch.Tensor, "num_ray num_sample"]] = None,
    ) -> RaySamples:
        """
        Samples points along rays.
        """
        if not importance_weights is None:
            assert not importance_t_samples is None, "Previous samples must be provided."
            t_samples = self.sample_along_rays_importance(
                importance_weights,
                importance_t_samples,
                num_sample,
            )
        else:
            t_samples = self.sample_along_rays_uniform(ray_bundle, num_sample)

        ray_samples = RaySamples(ray_bundle, t_samples)
        return ray_samples

    @jaxtyped(typechecker=typechecker)
    def sample_along_rays_uniform(
        self,
        ray_bundle: RayBundle,
        num_sample: int,
    ) -> Float[torch.Tensor, "num_ray num_sample"]:
        """
        Performs uniform sampling of points along rays.
        
        Args:
            ray_bundle: A ray bundle holding ray origins, directions, near and far bounds.
            num_sample: The number of samples to be generated along each ray.
        
        Returns:
            t_samples: The distance values sampled along rays. 
                The values should lie in the range defined by the near and
                far bounds of the ray bundle.
        """

        # parse ray bundle
        ray_origins = ray_bundle.origins
        num_ray = ray_origins.shape[0]
        device = ray_origins.device
        near = ray_bundle.nears[0].item()
        far = ray_bundle.fars[0].item()

        # ============================
        # Task 1-2. Implement Ray Sampling
        # DO NOT change the code outside this part.
        
        # ============================

        return t_samples

    @jaxtyped(typechecker=typechecker)
    def sample_along_rays_importance(
        self,
        weights: Float[torch.Tensor, "num_ray num_sample"],
        t_samples: Float[torch.Tensor, "num_ray num_sample"],
        num_sample: int,
    ) -> Float[torch.Tensor, "num_ray new_num_sample"]:
        """
        Performs the inverse CDF sampling of points along rays given weights
        indicating the 'importance' of each given sample.
        """

        # NOTE: The elements of 't_samples' are assumed to be ordered.
        t_mid = 0.5 * (t_samples[..., 1:] + t_samples[..., :-1])
        weights_mid = weights[..., 1:-1]
        new_t_samples = sample_pdf(t_mid, weights_mid, num_sample)

        # combine the new samples with the previous ones.
        # NOTE: The elements of 't_samples' must be sorted.
        t_samples = torch.cat([t_samples, new_t_samples], -1)
        t_samples, _ = torch.sort(t_samples, dim=-1)

        return t_samples

    @jaxtyped(typechecker=typechecker)
    def map_t_to_euclidean(
        self,
        t_values: Shaped[torch.Tensor, "..."],
        near: float,
        far: float,
    ) -> Shaped[torch.Tensor, "..."]:
        """
        Maps values in the parametric space [0, 1] to Euclidean space [near, far].
        """
        return near * (1.0 - t_values) + far * t_values

    @jaxtyped(typechecker=typechecker)
    def create_t_bins(
        self,
        num_bin: int,
        device: Union[int, torch.device],
    ) -> Float[torch.Tensor, "num_bin"]:
        """
        Generates samples of t's by subdividing the interval [0.0, 1.0] inclusively.
        """
        assert isinstance(num_bin, int), (
            f"Expected an integer for parameter 'num_samples'. Got a value of type {type(num_bin)}."
        )
        t_bins = torch.linspace(0.0, 1.0, num_bin, device=device)

        return t_bins
