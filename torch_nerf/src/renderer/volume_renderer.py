"""
Volume renderer implemented using Pytorch.
"""

import torch


class VolumeRenderer(object):
    """
    Volume renderer.
    """

    def __init__(self, integrator, sampler):
        """
        Constructor of class 'VolumeRenderer'.
        """
        self._integrator = integrator
        self._sampler = sampler

    def render_scene(
        self,
        target_scene,
        camera,
        pixel_indices,
        num_samples,
        project_to_ndc,
        weights=None,
        prev_t_samples=None,  # hierarchical sampling
        num_ray_batch=None,
    ):
        """
        Renders the scene by querying underlying 3D inductive bias.
        """

        ray_bundle = camera.generate_rays()
        ray_bundle.origins = ray_bundle.origins[pixel_indices, :]
        ray_bundle.directions = ray_bundle.directions[pixel_indices, :]
        ray_bundle.nears = ray_bundle.nears[pixel_indices]
        ray_bundle.fars = ray_bundle.fars[pixel_indices]

        # =====================================================================
        # Memory-bandwidth intensive operations (must be done directly on GPUs)
        # =====================================================================

        # sample points along rays
        ray_samples = self.sampler.sample_along_rays(
            ray_bundle,
            num_samples,
            importance_weights=weights,
            importance_t_samples=prev_t_samples,
        )

        # render rays
        pixel_rgb, weights, sigma, radiance = self.render_ray_batches(
            target_scene,
            ray_samples,
            num_batch=1 if num_ray_batch is None else num_ray_batch,
        )

        # =====================================================================
        # Memory-bandwidth intensive operations (must be done directly on GPUs)
        # =====================================================================

        return pixel_rgb, weights, ray_samples.t_samples

    def render_ray_batches(
        self,
        target_scene,
        ray_samples,
        num_batch,
    ):
        """
        Renders an image by dividing its pixels into small batches.
        """
        rgb = []
        weights = []
        sigma = []
        radiance = []

        sample_pts = ray_samples.compute_sample_coordinates()
        ray_dir = ray_samples.ray_bundle.directions
        delta_t = ray_samples.compute_deltas()

        # multiply ray direction norms to compute distance between
        # consecutive sample points
        dir_norm = torch.norm(ray_dir, dim=1, keepdim=True)
        delta_t = delta_t * dir_norm

        pts_chunks = torch.chunk(sample_pts, num_batch, dim=0)
        dir_chunks = torch.chunk(ray_dir, num_batch, dim=0)
        delta_chunks = torch.chunk(delta_t, num_batch, dim=0)
        assert len(pts_chunks) == len(dir_chunks) == len(delta_chunks), (
            f"{len(pts_chunks)} {len(dir_chunks)} {len(delta_chunks)}"
        )

        for pts_batch, dir_batch, delta_batch in zip(
            pts_chunks, dir_chunks, delta_chunks
        ):

            # query the scene to get density and radiance
            sigma_batch, radiance_batch = target_scene.query_points(pts_batch, dir_batch)

            # compute pixel colors by evaluating the volume rendering equation
            rgb_batch, weights_batch = self.integrator.integrate_along_rays(
                sigma_batch, radiance_batch, delta_batch
            )

            # collect rendering outputs
            rgb.append(rgb_batch)
            weights.append(weights_batch)
            sigma.append(sigma_batch)
            radiance.append(radiance_batch)

        pixel_rgb = torch.cat(rgb, dim=0)
        weights = torch.cat(weights, dim=0)
        sigma = torch.cat(sigma, dim=0)
        radiance = torch.cat(radiance, dim=0)

        return pixel_rgb, weights, sigma, radiance

    @property
    def integrator(self):
        return self._integrator

    @property
    def sampler(self):
        return self._sampler
