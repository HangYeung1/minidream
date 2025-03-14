from parser import Config

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from nerf import NeRF, Renderer


def sample_from_range(range: tuple[float, float]):
    """Uniformly sample a value from a given range."""
    return torch.rand(1) * (range[1] - range[0]) + range[0]


def train3d(config: Config):
    """Train a NeRF model with given configuration. Save in config.output_path.

    Arguments:
        config (Config): Configuration for training.
    """

    nerf = NeRF(128, 6).to(config.device)
    renderer = Renderer(
        config.render_dims,
        config.sample_range,
        config.render_samples,
        config.device,
        config.dtype,  # TODO: check decay
    )
    guide = config.guide(
        prompt=config.prompt,
        negative_prompt=config.negative_prompt,
        t_range=config.t_range,
        guidance_scale=config.guidance_scale,
        device=config.device,
        dtype=config.dtype,
    )
    optimizer = torch.optim.Adam(
        nerf.parameters(),
        lr=config.lr,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )

    for i in tqdm(range(config.iterations), "Training model..."):
        # Render random image
        theta = sample_from_range(config.theta_range)
        phi = sample_from_range(config.phi_range)
        focal = (sample_from_range(config.focal_range) * config.render_dims[1]).item()
        radius = sample_from_range(config.radius_range).item()

        pose = renderer.get_pose(theta, phi, radius).to(config.device)
        rgb = renderer.render(nerf, pose, focal)[0].permute(2, 0, 1).unsqueeze(0)

        # Optimize representation
        optimizer.zero_grad()
        loss = guide.calculate_sds_loss(rgb, theta, phi)
        loss.backward()
        optimizer.step()

        # Save weights and images
        if i % config.output_interval == 0 or i == config.iterations - 1:
            torch.save(nerf.state_dict(), config.output_path / "weights" / f"{i}.pth")
            with torch.no_grad():
                test_rgb, test_depth = renderer.render(
                    nerf,
                    renderer.get_pose(
                        0,
                        75,
                        sum(config.radius_range) / 2,
                    ).to(config.device),
                    sum(config.focal_range) / 2 * config.render_dims[1],
                )

                test_rgb = test_rgb.cpu()
                test_depth = test_depth.cpu()
                depth_display = (test_rgb.mean(2) > 0.03) * test_depth

                plt.figure()
                plt.subplot(1, 2, 1)
                plt.imshow(test_rgb)
                plt.subplot(1, 2, 2)
                plt.imshow(depth_display, cmap="gray")
                plt.savefig(config.output_path / "images" / f"{i}.png")
                plt.close()
