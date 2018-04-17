import models
import argparse
import tensorflow as tf

def parse_args():
    parser = argparse.ArgumentParser(description="Up-Scales an image using Image Super Resolution Model")
    parser.add_argument("imgpath", type=str, nargs="+", help="Path to input image")
    args = parser.parse_args()

    paths = args.imgpath
    return paths

def super_resolve(path):
    with tf.device('/CPU:0'):
        scale_factor = 2
        model = models.DistilledResNetSR(scale_factor)
        model.upscale(path, save_intermediate=True, mode="patch", patch_size=8, suffix="scaled")


if __name__ == "__main__":
    paths = parse_args()
    super_resolve(paths[0])