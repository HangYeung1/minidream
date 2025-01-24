from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from guidance import IFGuide
from nerf import NeRF, Renderer

torch.manual_seed(42)
torch.cuda.manual_seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

height = width = 64
prompt = "an apple"
negative_prompt = "cropped, out of frame, morbid, mutilated, bizarre, corrupted, malformed, low quality, artifacts, watermark, signature"
interval = 250

num_samples = 64
render_dist = 4.0
near = 2.0
far = 8.0

lr = 1e-4
eps = 1e-4
weight_decay = 0.1
guidance = 100

save_name = "apple6"
save_path = "output"

save_path = Path(save_path) / save_name
weights_path = save_path / "weights"
images_path = save_path / "images"
weights_path.mkdir(parents=True, exist_ok=True)
images_path.mkdir(parents=True, exist_ok=True)


def train3d(iterations):
    nerf = NeRF(128, 6).to(device)
    renderer = Renderer(64, 64, near, far, num_samples, device, torch.float32)
    guide = IFGuide(
        t_range=(0.02, 0.98),
        guidance_scale=guidance,
        device=device,
        dtype=torch.float16,
    )
    optimizer = torch.optim.Adam(
        nerf.parameters(), lr=lr, eps=eps, weight_decay=weight_decay
    )

    pos_embeds_top = guide.encode_text("overhead view of " + prompt)
    pos_embeds_front = guide.encode_text("front view of " + prompt)
    pos_embeds_side = guide.encode_text("side view of " + prompt)
    pos_embeds_back = guide.encode_text("back view of " + prompt)

    neg_embeds = guide.encode_text(negative_prompt)

    for i in tqdm(range(iterations), "Optimizing model..."):
        theta = torch.rand(1) * 360
        phi = torch.rand(1) * 100
        focal = ((torch.rand(1) * 0.65 + 0.7) * width).item()

        pose = renderer.get_pose(theta, phi, render_dist).to(device)
        rgb = renderer.render(nerf, pose, focal)[0].permute(2, 0, 1).unsqueeze(0).half()

        # Get positionally modulated prompts
        curr_embeds = None
        if phi < 60:
            curr_embeds = pos_embeds_top
        elif theta <= 45 and theta > 315:
            curr_embeds = pos_embeds_front
        elif theta > 135 and theta < 225:
            curr_embeds = pos_embeds_back
        else:
            curr_embeds = pos_embeds_side

        optimizer.zero_grad()
        loss = guide.calculate_sds_loss(rgb, curr_embeds, neg_embeds)
        loss.backward()
        optimizer.step()

        if i % interval == 0:
            torch.save(nerf.state_dict(), weights_path / f"ckpt_{i}.pth")
            with torch.no_grad():
                test_rgb, test_depth = renderer.render(
                    nerf, renderer.get_pose(0, 75, render_dist).to(device), width
                )

                test_rgb = test_rgb.cpu()
                test_depth = test_depth.cpu()
                depth_display = (test_rgb.mean(2) > 0.03) * test_depth

                plt.figure()
                plt.subplot(1, 2, 1)
                plt.imshow(test_rgb)
                plt.subplot(1, 2, 2)
                plt.imshow(depth_display, cmap="gray")
                plt.savefig(images_path / f"pic_{i}.png")
                plt.close()

    torch.save(nerf.state_dict(), weights_path / "ckpt_final.pth")


if __name__ == "__main__":
    train3d(30000)
