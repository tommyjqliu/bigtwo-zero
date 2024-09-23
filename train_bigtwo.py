import os

from train.dmc import train
from train.arguments import parser

if __name__ == '__main__':
    flags = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu_devices
    train(flags)
