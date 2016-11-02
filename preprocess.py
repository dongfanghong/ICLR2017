from PIL import Image
import glob
import time
import argparse
import os
from multiprocessing import Pool
import scipy.misc


def check_gray_rgba(input_file):
    """
    Checks if the image is grayscale or rgba and converts to RGB if necessary.
    """
    gray = False
    rgba = False
    img = scipy.misc.imread(input_file)
    if len(img.shape) < 3:
        gray = True
    elif img.shape[2] == 4:
        rgba = True
    img = Image.open(input_file)
    if gray or rgba:
        img_rgb = Image.new('RGB', img.size)
        img_rgb.paste(img)
        img = img_rgb
    return img


def check_image_size(img, box):
    """
    Resizes the image if it's too small or too big.
    """
    # Check if image is too small
    if min(img.size) < max(box):
        factor = 1
        min_size = min(img.size)
        while min_size * factor < max(box):
            factor += 1
        img = img.resize((img.size[0] * factor, img.size[1] * factor),
                         Image.BILINEAR)
    # Preresize image with factor 2, 4, 8 and fast algorithm
    factor = 1
    while (img.size[0] / factor > 2 * box[0] and
           img.size[1] / factor > 2 * box[1]):
        factor *= 2
    if factor > 1:
        img.thumbnail((img.size[0] / factor, img.size[1] / factor),
                      Image.NEAREST)
    return img


def crop_image(img, box, fit):
    """
    Computes cropping box and gets the cropped part.
    """
    if fit:
        x1 = y1 = 0
        x2, y2 = img.size
        wRatio = 1.0 * x2 / box[0]
        hRatio = 1.0 * y2 / box[1]
        if hRatio > wRatio:
            y1 = int(y2 / 2 - box[1] * wRatio / 2)
            y2 = int(y2 / 2 + box[1] * wRatio / 2)
        else:
            x1 = int(x2 / 2 - box[0] * hRatio / 2)
            x2 = int(x2 / 2 + box[0] * hRatio / 2)
    return img.crop((x1, y1, x2, y2))


def resize_and_crop_image(input_file, output_file,
                          pixel_size=64, fit=True):
    """
    Downsample the image.
    """
    img = check_gray_rgba(input_file)
    box = (pixel_size, pixel_size)
    img = check_image_size(img, box)
    img = crop_image(img, box, fit)
    img = img.resize(box, Image.ANTIALIAS)
    with open(output_file, 'wb') as out:
        img.save(out, 'JPEG', quality=95)


def process_image(idx, in_fn, t0):
    """
    Processes one image.
    """
    out_fn = os.path.join(args.out_dir, os.path.basename(in_fn))
    resize_and_crop_image(in_fn, out_fn, pixel_size=args.pixel_size)
    if (idx + 1) % 1000 == 0:
        t1 = time.time()
        print 'Saved {} images in {} seconds'.format(idx + 1, t1 - t0)


def process_images():
    """
    Processes folder of images and saves them.
    """
    t0 = time.time()
    pool = Pool(processes=args.threads)
    for idx, in_fn in enumerate(glob.glob(args.input_dir + '*')):
        pool.apply_async(func=process_image, args=(idx, in_fn, t0))
    pool.close()
    pool.join()
    t1 = time.time()
    print 'Saved {} images in {} seconds'.format(idx + 1, t1 - t0)


if __name__ == '__main__':
    """
    For example, run:
    python preprocess.py /atlas/u/nj/imagenet/test_orig/
        /atlas/u/nj/imagenet/test_small/ 64 --threads 16
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_dir', help='The directory with images to be processed')
    parser.add_argument(
        'out_dir', help='The directory to save processed images in')
    parser.add_argument(
        'pixel_size', help='The pixel side length after processing', type=int,
        default=64)
    parser.add_argument(
        '--threads', help='The number of threads to use', type=int,
        default=1)
    args = parser.parse_args()
    process_images()
