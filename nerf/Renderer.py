import cv2
import numpy as np
import torch
import torch.nn as nn

from .NeRF import NeRF


class Renderer(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        near: float,
        far: float,
        num_samples: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """Initialize rendering charateristics."""

        super().__init__()

        self.height = height
        self.width = width
        self.near = near
        self.far = far
        self.num_samples = num_samples
        self.device = device
        self.dtype = dtype

    @torch.no_grad
    def get_pose(
        self,
        theta_deg: float | torch.Tensor,
        phi_deg: float | torch.Tensor,
        radius: float | torch.Tensor,
    ):
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

    @torch.no_grad
    def get_rays(self, focal: float, pose: torch.Tensor):
        """Get raymarching info."""

        # Make pixel grid
        x, y = torch.meshgrid(
            torch.arange(self.width, dtype=torch.float32, device=self.device),
            torch.arange(self.height, dtype=torch.float32, device=self.device),
            indexing="xy",
        )

        # Calculate raymarch direction for each pixel
        directions = (
            torch.stack(
                [
                    (x - self.width / 2) / focal,
                    -(y - self.height / 2) / focal,
                    -torch.ones_like(x, device=self.device),
                ],
                dim=-1,
            )
            @ pose[:3, :3].T
        )
        origin = pose[:3, -1]

        return origin, directions

    def render(self, nerf: NeRF, pose: torch.Tensor, focal: float):
        """Render image given position."""

        # Get random t's from intervals
        bins = torch.linspace(
            self.near,
            self.far,
            self.num_samples + 1,
            dtype=self.dtype,
            device=self.device,
        )
        bin_size = bins[1] - bins[0]
        t = bins[:-1] + bin_size * torch.rand(
            (self.height, self.width, self.num_samples),
            dtype=self.dtype,
            device=self.device,
        )

        # Get point info from each ray
        # Janky chunk to prevent another bluescreen
        origin, directions = self.get_rays(focal, pose)
        points = origin + directions.unsqueeze(-2) * t.unsqueeze(-1)

        num_chunks = int(points.numel() / (3 * 2**15))
        points_chunks = points.view(-1, 3).chunk(num_chunks)
        rgb_s_chunks = [nerf(chunk) for chunk in points_chunks]
        rgb_s = torch.cat(rgb_s_chunks).view(
            self.height, self.width, self.num_samples, 4
        )

        rgb = rgb_s[..., :3]
        sigmas = rgb_s[..., -1]

        # Calculate attenuation at each point
        inf = 1e9
        deltas = torch.cat(
            (
                t[..., 1:] - t[..., :-1],
                torch.full_like(t[..., :1], inf, dtype=self.dtype, device=self.device),
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
        rgb_matrix = torch.sum(weights.unsqueeze(-1) * rgb, dim=-2)
        depth_matrix = torch.sum(weights * t, dim=-1)

        return rgb_matrix, depth_matrix

    @torch.no_grad
    def render_video(
        self,
        nerf: NeRF,
        output_path: str,
        focal: float,
        frame_rate: int = 10,
    ):
        """Save 360 video of nerf."""
        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"H264"),
            frame_rate,
            (self.width, self.height),
        )

        for i in range(0, 360, 5):
            pose = self.get_pose(i, 75, 3.5)
            rgb, _ = self.render(nerf, pose, focal)

            out.write(
                cv2.cvtColor(
                    (rgb * 255).cpu().numpy().astype(np.uint8),
                    cv2.COLOR_RGB2BGR,
                )
            )

        out.release()
