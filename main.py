import models
import argparse
import tensorflow as tf
import base64


def parse_args():
    parser = argparse.ArgumentParser(description="Up-Scales an image using Image Super Resolution Model")
    parser.add_argument("imgpath", type=str, nargs="+", help="Path to input image")
    args = parser.parse_args()
    paths = args.imgpath
    return paths

def super_resolve(path):
    with tf.device('/CPU:0'):
        model = models.DistilledResNetSR(scale_factor=2)
        model.upscale(path, save_intermediate=True, mode="patch", patch_size=8, suffix="scaled")


def web_request(base64_img):
    with open('input_image', 'wb') as f:
        f.write(base64.decodebytes(base64_img))

    super_resolve("input_image")

    with open("input_image_scaled(2x)", "rb") as f:
        encoded_string = base64.b64encode(f.read())

    return encoded_string


if __name__ == "__main__":
    paths = parse_args()
    super_resolve(paths[0])