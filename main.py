from parser import parse_args

import torch

from train3d import train3d

torch.manual_seed(42)
torch.cuda.manual_seed(42)


def main():
    args = parse_args()
    train3d(args)


if __name__ == "__main__":
    main()
