import argparse
import numpy as np


def make_parser():
    parser = argparse.ArgumentParser("MieMieGAN Demo!")
    parser.add_argument(
        "--param_path",
        default="",
        type=str,
        help="param_path",
    )
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()
    param_path = args.param_path
    param_path = '06.param'

    lines = []
    with open(param_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            lines.append(line)
    tensors_dic = {}
    tensor_id = 0
    for i, line in enumerate(lines):
        if i < 2:
            pass
        else:
            ss = line.split()
            in_num = int(ss[2])
            out_num = int(ss[3])
            p = 4
            for i1 in range(in_num):
                tensor_name = ss[p]
                if tensor_name not in tensors_dic.keys():
                    aaaaaaaaaa = 'tensor_%.8d' % (tensor_id, )
                    tensor_id += 1
                    tensors_dic[tensor_name] = aaaaaaaaaa
                p += 1
            for i2 in range(out_num):
                tensor_name = ss[p]
                if tensor_name not in tensors_dic.keys():
                    aaaaaaaaaa = 'tensor_%.8d' % (tensor_id, )
                    tensor_id += 1
                    tensors_dic[tensor_name] = aaaaaaaaaa
                p += 1
    content = ''
    for i, line in enumerate(lines):
        if i < 2:
            content += line + '\n'
        else:
            ss = line.split()
            in_num = int(ss[2])
            out_num = int(ss[3])
            p = 4 + in_num + out_num - 1
            for i1 in range(in_num):
                tensor_name = ss[p]
                if tensor_name != 'images':
                    line = line.replace(tensor_name, tensors_dic[tensor_name])
                p -= 1
            for i2 in range(out_num):
                tensor_name = ss[p]
                if tensor_name != 'images':
                    line = line.replace(tensor_name, tensors_dic[tensor_name])
                p -= 1
            content += line + '\n'
    with open(param_path, 'w', encoding='utf-8') as f:
        f.write(content)
        f.close()





