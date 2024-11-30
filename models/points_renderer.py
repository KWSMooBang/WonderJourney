import torch

from torch import nn


class PointsRenderer(nn.Module):
    def __init__(self, rasterizer, compositor) -> None:
        super().__init__()
        self.rasterizer = rasterizer
        self.compostior = compositor

    def forward(self, point_clouds, return_z=False, return_bg_mask=False, return_fragment_idx=False, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(point_clouds, **kwargs)

        r = self.rasterizer.raster_settings.radius

        zbuf = fragments.zbuf.permute(0, 3, 1, 2)
        fragment_idx = fragments.idx.long().permute(0, 3, 1, 2)
        background_mask = fragment_idx[:, 0] < 0
        images = self.compostior(
            fragment_idx,
            zbuf,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )

        # permute so iamge comes at the end
        images = images.permute(0, 2, 3, 1)

        ret = [images]
        if return_z:
            ret.append(fragments.zbuf)
        if return_bg_mask:
            ret.append(background_mask)
        if return_fragment_idx:
            ret.append(fragments.idx.long())

        if len(ret) == 1:
            ret = images
        return ret