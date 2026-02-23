"""
Utilities related to sampling.
"""

from beartype import beartype as typechecker

from jaxtyping import Float, jaxtyped
import torch


@jaxtyped(typechecker=typechecker)
def sample_pdf(
    bins: Float[torch.Tensor, "num_ray num_bin"],
    weights: Float[torch.Tensor, "num_ray num_weight"],
    num_sample: int,
) -> Float[torch.Tensor, "num_ray num_sample"]:
    """
    Draws samples from the probability density represented by given weights.
    """
    assert bins.shape[1] == weights.shape[1] + 1, f"{bins.shape[1]}, {weights.shape[1]}"

    # construct the PDF
    weights += 1e-5
    normalizer = torch.sum(weights, dim=-1, keepdim=True)
    pdf = weights / normalizer

    # compute the CDF
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    # sample from the uniform distribution: U[0, 1)
    cdf_ys = torch.rand(list(cdf.shape[:-1]) + [num_sample], device=cdf.device)
    cdf_ys = cdf_ys.contiguous()

    # inverse CDF sampling
    indices = torch.searchsorted(cdf, cdf_ys, right=True)
    lower = torch.max(torch.zeros_like(indices - 1), indices - 1)
    upper = torch.min(cdf.shape[-1] - 1 * torch.ones_like(indices), indices)

    cdf_lower = torch.gather(cdf, 1, lower)
    cdf_upper = torch.gather(cdf, 1, upper)
    bins_lower = torch.gather(bins, 1, lower)
    bins_upper = torch.gather(bins, 1, upper)

    # approximate the CDF to a linear function within each interval
    denom = cdf_upper - cdf_lower
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (cdf_ys - cdf_lower) / denom  # i.e., cdf_y = cdf_lower + t * (cdf_upper - cdf_lower)
    t_samples = bins_lower + t * (bins_upper - bins_lower)

    assert torch.isnan(t_samples).sum() == 0

    return t_samples
