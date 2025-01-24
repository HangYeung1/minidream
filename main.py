from parser import parse_args

import torch

torch.manual_seed(42)
torch.cuda.manual_seed(42)


def main():
    args = parse_args()


if __name__ == "__main__":
    main()
