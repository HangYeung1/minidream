import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import yaml

from guidance import GuideMap


@dataclass
class Config:
    """Diffusion Configuration Data Class.

    Attributes:
        prompt (str): Guiding prompt.
        negative_prompt (str): Negative prompt.
        guide (GuideType): Guidance model class.
        iterations (int): Number of training iterations.
        lr (float): Adam learning rate.
        eps (float): Adam epsilon.
        weight_decay (float): Adam weight decay rate.
        guidance_scale (float): Classifier-free guidance weight.
        t_range (Tuple[float, float]): Diffusion sampling interval.
        theta_range (Tuple[float, float]): Render azimuth interval.
        phi_range (Tuple[float, float]): Render zenith interval.
        focal_range (Tuple[float, float]): Render focal length interval.
        radius_range (Tuple[float, float]): Render radius interval.
        sample_range (Tuple[float, float]): Render sample interval.
        train_render_dims (Tuple[int, int]): Training render resolution.
        train_render_samples (int): Number of samples per training render.
        check_render_dims (Tuple[int, int]): Checkpoint render resolution.
        check_render_samples (int): Number of samples per checkpoint render.
        check_frame_rate (int): Video render frame rate.
        output_path (Path): File output path.
        output_interval (int): Iterations between weight checkpoint.
        device (torch.device): Device of training.
        dtype (torch.dtype): Precision of training.
    """

    prompt: str
    negative_prompt: str
    guide: type
    iterations: int
    lr: float
    eps: float
    weight_decay: float
    guidance_scale: float
    t_range: Tuple[float, float]
    theta_range: Tuple[float, float]
    phi_range: Tuple[float, float]
    focal_range: Tuple[float, float]
    radius_range: Tuple[float, float]
    sample_range: Tuple[float, float]
    train_render_dims: Tuple[int, int]
    train_render_samples: int
    check_render_dims: Tuple[int, int]
    check_render_samples: int
    check_frame_rate: int
    output_path: Path
    output_interval: int
    device: torch.device
    dtype: torch.dtype


def yaml_to_args(yaml_path: str) -> List[str]:
    """Convert YAML file to argparse argument list."""

    with open(yaml_path) as file:
        yaml_dict = yaml.safe_load(file)
        yaml_list = []
        for key, value in yaml_dict.items():
            if key == "yaml":
                raise argparse.ArgumentTypeError(
                    "YAML file cannot include another YAML file."
                )
            yaml_list.append(str(f"--{key}"))
            yaml_list += (
                [str(val) for val in value] if isinstance(value, list) else [str(value)]
            )
        return yaml_list


def parse_args(arg_list: None | List[str] = None) -> Config:
    """Parse SDS configuration from argparse argument list.

    Arguments:
        arg_list (List[str], optional): Argument list to parse. Default to CLI.

    Returns:
        Converted Config data class.
    """

    # Make parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--prompt", type=str)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--guide", type=str, choices=GuideMap.keys(), default="IFGuide")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--guidance_scale", type=float, default=100)
    parser.add_argument("--t_range", type=float, nargs=2, default=[0.02, 0.98])
    parser.add_argument("--theta_range", type=float, nargs=2, default=[0, 360])
    parser.add_argument("--phi_range", type=float, nargs=2, default=[0, 100])
    parser.add_argument("--focal_range", type=float, nargs=2, default=[0.7, 1.35])
    parser.add_argument("--radius_range", type=float, nargs=2, default=[4, 4.15])
    parser.add_argument("--sample_range", type=float, nargs=2, default=[2, 8])
    parser.add_argument("--train_render_dims", type=int, nargs=2, default=[64, 64])
    parser.add_argument("--train_render_samples", type=int, default=64)
    parser.add_argument("--check_render_dims", type=int, nargs=2, default=[128, 128])
    parser.add_argument("--check_render_samples", type=int, default=128)
    parser.add_argument("--check_frame_rate", type=int, default=12)
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument("--output_interval", type=int, default=250)

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "float32", "float64"],
        default="float16" if torch.cuda.is_available() else "float32",
    )

    parser.add_argument("--yaml", type=str)

    # Parse arguments (with YAML support)
    args = parser.parse_args(args=arg_list)
    if args.yaml:
        yaml_arg_list = yaml_to_args(args.yaml)
        args = parser.parse_args(args=yaml_arg_list)
    if not args.prompt:
        raise argparse.ArgumentTypeError("Prompt must be provided.")

    # Make save directories
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True)
    (output_path / "weights").mkdir()
    (output_path / "images").mkdir()

    # Dump config to yaml
    with open(output_path / "config.yaml", "w") as file:
        args.__dict__.pop("yaml", None)
        yaml.dump(
            args.__dict__,
            file,
            width=None,
            sort_keys=False,
            default_flow_style=None,
        )

    return Config(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        guide=GuideMap[args.guide],
        iterations=args.iterations,
        lr=args.lr,
        eps=args.eps,
        weight_decay=args.weight_decay,
        guidance_scale=args.guidance_scale,
        t_range=tuple(args.t_range),
        theta_range=tuple(args.theta_range),
        phi_range=tuple(args.phi_range),
        focal_range=tuple(args.focal_range),
        radius_range=tuple(args.radius_range),
        sample_range=tuple(args.sample_range),
        train_render_dims=tuple(args.train_render_dims),
        train_render_samples=args.train_render_samples,
        check_render_dims=tuple(args.check_render_dims),
        check_render_samples=args.check_render_samples,
        check_frame_rate=args.check_frame_rate,
        output_path=Path(args.output_path),
        output_interval=args.output_interval,
        device=torch.device(args.device),
        dtype=getattr(torch, args.dtype),
    )


# Run parser.py to test the parser
if __name__ == "__main__":
    print(parse_args())
