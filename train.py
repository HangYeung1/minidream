from parser import Config

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from nerf import NeRF


def sample_from_range(range: tuple[float, float]) -> torch.Tensor:
    """Uniformly sample a value from a given range."""
    return torch.rand(1) * (range[1] - range[0]) + range[0]


def train(config: Config) -> None:
    """Train a NeRF model with given configuration. Save in config.output_path.

    Arguments:
        config (Config): Configuration for training.
    """

    nerf = NeRF(
        config.render_dims,
        config.sample_range,
        config.render_samples,
        config.device,
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

    for i in tqdm(range(1, config.iterations + 1), "Training model..."):
        # Render random image
        theta = sample_from_range(config.theta_range).item()
        phi = sample_from_range(config.phi_range).item()
        radius = sample_from_range(config.radius_range).item()
        focal = (sample_from_range(config.focal_range) * config.render_dims[1]).item()

        rgb = (
            nerf.render(theta, phi, radius, focal)[0]
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(config.dtype)
        )

        # Optimize representation
        optimizer.zero_grad()
        loss = guide.calculate_sds_loss(rgb, theta, phi)
        loss.backward()
        optimizer.step()

        # Save weights and images
        if i % config.output_interval == 0 or i == config.iterations:
            torch.save(nerf.state_dict(), config.output_path / "weights" / f"{i}.pth")
            with torch.no_grad():
                test_rgb, test_depth = nerf.render(
                    0,
                    75,
                    sum(config.radius_range) / 2,
                    sum(config.focal_range) / 2 * config.render_dims[1],
                )

                test_rgb = test_rgb.cpu()
                test_depth = test_depth.cpu()
                depth_display = (test_rgb.mean(2) > 0.03) * test_depth

                plt.figure()
                plt.subplot(1, 2, 1)
                plt.axis("off")
                plt.title("RGB")
                plt.imshow(test_rgb)
                plt.subplot(1, 2, 2)
                plt.axis("off")
                plt.title("Depth")
                plt.imshow(depth_display, cmap="gray")
                plt.savefig(config.output_path / "images" / f"{i}.png")
                plt.close()

    # Render final video
    nerf.render_video(
        str(config.output_path / "video.mp4"),
        10,
        sum(config.radius_range) / 2,
        sum(config.focal_range) / 2 * config.render_dims[1],
    )
