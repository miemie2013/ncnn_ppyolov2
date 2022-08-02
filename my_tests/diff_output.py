import argparse
import numpy as np


def make_parser():
    parser = argparse.ArgumentParser("MieMieGAN Demo!")
    parser.add_argument(
        "--ncnn_output",
        default="",
        type=str,
        help="ncnn_output",
    )
    parser.add_argument(
        "--torch_output",
        default="",
        type=str,
        help="torch_output",
    )
    parser.add_argument(
        "--d_value",
        default=0.00001,
        type=float,
        help="d_value",
    )
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()
    ncnn_output = args.ncnn_output
    torch_output = args.torch_output

    dic2 = np.load(torch_output)
    y2 = dic2['y']

    with open(ncnn_output, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
    line = line[:-1]
    ss = line.split(',')
    y = []
    for s in ss:
        y.append(float(s))
    y = np.array(y).astype(np.float32)
    y = np.reshape(y, y2.shape)
    print(y2.shape)

    ddd = np.sum((y - y2) ** 2)
    print('ddd=%.9f' % ddd)





