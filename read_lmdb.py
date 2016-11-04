# from https://github.com/fyu/lsun/blob/master/data.py
from __future__ import print_function
import argparse
import cv2
import lmdb
import numpy
import os
from os.path import exists, join
from PIL import Image
import time
from multiprocessing import Pool

__author__ = 'Fisher Yu'
__email__ = 'fy@cs.princeton.edu'
__license__ = 'MIT'


def view(db_path):
    print('Viewing', db_path)
    print('Press ESC to exist or SPACE to advance.')
    window_name = 'LSUN'
    cv2.namedWindow(window_name)
    env = lmdb.open(db_path, map_size=1099511627776,
                    max_readers=100, readonly=True)
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, val in cursor:
            print('Current key:', key)
            img = cv2.imdecode(
                numpy.fromstring(val, dtype=numpy.uint8), 1)
            cv2.imshow(window_name, img)
            c = cv2.waitKey()
            if c == 27:
                break


def export_image(count, key, val, out_dir, flat, t0):
    if not flat:
        image_out_dir = join(out_dir, '/'.join(key[:6]))
    else:
        image_out_dir = out_dir
    if not exists(image_out_dir):
        os.makedirs(image_out_dir)
    image_out_path = join(image_out_dir, key + '.webp')
    with open(image_out_path, 'w') as fp:
        fp.write(val)
    # Convert from webp to jpeg
    img = Image.open(image_out_path).convert('RGB')
    out_fn = image_out_path[:-4] + 'JPEG'
    img.save(out_fn, 'JPEG')
    # Remove webp
    os.remove(image_out_path)
    if count % 1000 == 0:
        t1 = time.time()
        print('Finished', count, 'images in {} seconds'.format(t1-t0))


def export_images(db_path, out_dir, flat=False, limit=-1):
    print('Exporting', db_path, 'to', out_dir)
    env = lmdb.open(db_path, map_size=1099511627776,
                    max_readers=100, readonly=True)
    count = 0
    t0 = time.time()
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        pool = Pool(processes=16)
        for key, val in cursor:
            count += 1
            if count > 30000:
                break
            pool.apply_async(
                func=export_image, args=(count, key, val, out_dir, flat, t0))
        pool.close()
        pool.join()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', nargs='?', type=str,
                        choices=['view', 'export'],
                        help='view: view the images in the lmdb database '
                             'interactively.\n'
                             'export: Export the images in the lmdb databases '
                             'to a folder. The images are grouped in subfolders'
                             ' determinted by the prefiex of image key.')
    parser.add_argument('lmdb_path', nargs='+', type=str,
                        help='The path to the lmdb database folder. '
                             'Support multiple database paths.')
    parser.add_argument('--out_dir', type=str, default='')
    parser.add_argument('--flat', action='store_true',
                        help='If enabled, the images are imported into output '
                             'directory directly instead of hierarchical '
                             'directories.')
    args = parser.parse_args()

    command = args.command
    lmdb_paths = args.lmdb_path

    for lmdb_path in lmdb_paths:
        if command == 'view':
            view(lmdb_path)
        elif command == 'export':
            export_images(lmdb_path, args.out_dir, args.flat)


if __name__ == '__main__':
    main()