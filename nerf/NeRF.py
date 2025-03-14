from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from .NeRFNetwork import NeRFNetwork


class NeRF(nn.Module):
    def __init__(
        self,
        sample_range: Tuple[float, float],
        device: torch.device,
    ):
        """Initialize rendering characteristics."""

        super().__init__()

        self.sample_range = sample_range
        self.device = device

        self.network = NeRFNetwork().to(self.device)

    @torch.no_grad()
    def get_pose(
        self,
        theta_deg: float,
        phi_deg: float,
        radius: float,
    ) -> torch.Tensor:
        """Calculate OpenGL style camera pose."""

        theta_rad = torch.deg2rad(torch.tensor(theta_deg))
        phi_rad = torch.deg2rad(torch.tensor(phi_deg))

        # Calculate position vector
        x = radius * torch.sin(phi_rad) * torch.cos(theta_rad)
        y = radius * torch.cos(phi_rad)
        z = radius * torch.sin(phi_rad) * torch.sin(theta_rad)
        position = torch.tensor([x, y, z])

        # Calculate rotation matrix
        forward = position
        forward = forward / torch.norm(forward)
        right = torch.linalg.cross(torch.tensor([0.0, 1.0, 0.0]), forward)
        right = right / torch.norm(right)
        up = torch.linalg.cross(forward, right)
        rotation = torch.stack([right, up, forward], dim=1)

        # Combine to pose
        pose = torch.cat([rotation, position.unsqueeze(1)], dim=1)
        return pose

    @torch.no_grad()
    def get_rays(
        self, focal: float, pose: torch.Tensor, dims: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get raymarching info."""

        # Make pixel grid
        x, y = torch.meshgrid(
            torch.arange(dims[1], device=self.device),
            torch.arange(dims[0], device=self.device),
            indexing="xy",
        )

        # Calculate raymarch direction for each pixel
        directions = (
            torch.stack(
                [
                    (x - dims[1] / 2) / focal,
                    -(y - dims[0] / 2) / focal,
                    -torch.ones_like(x, device=self.device),
                ],
                dim=-1,
            )
            @ pose[:3, :3].T
        )
        origin = pose[:3, -1]

        return origin, directions

    def render(
        self,
        theta: float,
        phi: float,
        radius: float,
        focal: float,
        dims: Tuple[int, int],
        num_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Render image given position."""

        pose = self.get_pose(theta, phi, radius).to(self.device)

        # Get random t's from intervals
        bins = torch.linspace(
            self.sample_range[0],
            self.sample_range[1],
            num_samples + 1,
            device=self.device,
        )
        bin_size = bins[1] - bins[0]
        t = bins[:-1] + bin_size * torch.rand(
            (dims[0], dims[1], num_samples),
            device=self.device,
        )

        # Get point info from each ray
        # Chunk to save memory
        origin, directions = self.get_rays(focal, pose, dims)
        points = origin + directions.unsqueeze(-2) * t.unsqueeze(-1)

        num_chunks = int(points.numel() / (3 * 2**15))
        points_chunks = points.view(-1, 3).chunk(num_chunks)
        rgb_s_chunks = [self.network(chunk) for chunk in points_chunks]
        rgb_s = torch.cat(rgb_s_chunks).view(dims[0], dims[1], num_samples, 4)

        rgb = rgb_s[..., :3]
        sigmas = rgb_s[..., -1]

        # Calculate attenuation at each point
        inf = 1e9
        deltas = torch.cat(
            (
                t[..., 1:] - t[..., :-1],
                torch.full_like(t[..., :1], inf, device=self.device),
            ),
            dim=-1,
        )
        sig_dels = torch.exp(-sigmas * deltas)
        alpha = 1.0 - sig_dels

        eps = 1e-9
        transmits_off = torch.cumprod((sig_dels + eps), -1)
        transmits = torch.roll(transmits_off, 1, -1)
        transmits[..., 0] = 1.0
        weights = alpha * transmits

        # Weigh color/depth contribution
        rgb_matrix = torch.clip(torch.sum(weights.unsqueeze(-1) * rgb, dim=-2), max=1.0)
        depth_matrix = torch.sum(weights * t, dim=-1)

        return rgb_matrix, depth_matrix

    @torch.no_grad()
    def render_video(
        self,
        output_path: str,
        frame_rate: int,
        radius: float,
        focal: float,
        dims: Tuple[int, int],
        num_samples: int,
    ) -> None:
        """Save five second 360 video of nerf."""
        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"avc1"),
            frame_rate,
            dims,
        )

        assert 360 % (frame_rate * 5) == 0
        for i in range(0, 360, int(360 // (frame_rate * 5))):
            rgb, _ = self.render(i, 75, radius, focal, dims, num_samples)
            out.write(
                cv2.cvtColor(
                    (rgb * 255).cpu().numpy().astype(np.uint8),
                    cv2.COLOR_RGB2BGR,
                )
            )

        out.release()
