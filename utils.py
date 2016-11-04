import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
import os
import fnmatch

#############################################################

pp = pprint.PrettyPrinter()
# get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

#############################################################


def imread(path):
    """
    Reads an image into np.float array.
    """
    return scipy.misc.imread(path).astype(np.float)


def center_crop(x, crop_h, crop_w=None, resize_w=64):
    """
    Crops the center patch of an image and resizes it.
    """
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    crop = scipy.misc.imresize(
        x[j:j + crop_h, i:i + crop_w], [resize_w, resize_w])
    return crop


def transform(image, npx=64, is_crop=True):
    """
    Either crops an image or returns the original image with pixels scaled to
    be between 0.0-1.0.
    Args:
        image: The image to be cropped/returned
        npx: # of pixels width/height of image
        is_crop: Whether or not to crop the image.
    """
    if is_crop:
        cropped_image = center_crop(image, npx)
    else:
        cropped_image = image
    return np.array(cropped_image) / 127.5 - 1.0


def get_image(image_path, image_size, is_crop=True):
    return transform(imread(image_path), image_size, is_crop)


def get_patches(image_path, down_sample_level=6):
    return img2downsampled(
        imread(image_path), down_sample_level, smooth=False)
    # return img2noise(imread(image_path), noise_ratio=0.25)
    # return img2patches(imread(image_path),32)


def get_patches_batch(filenames, get_patches, down_sample_level):
    """
    Puts all of the patches for a batch together in an array.
    """
    patch1_batch = []
    patch2_batch = []
    for filename in filenames:
        patch1, patch2 = get_patches(filename, down_sample_level)
        patch1_batch.append(patch1)
        patch2_batch.append(patch2)
    patch1_batch_array = np.array(patch1_batch, dtype='float32')
    patch2_batch_array = np.array(patch2_batch, dtype='float32')
    return patch1_batch_array, patch2_batch_array


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img


def imsave(images, size, path):
    """
    Saves images.
    """
    return scipy.misc.imsave(path, merge(images, size))


def save_images(images, size, image_path):
    """
    """
    return imsave(inverse_transform(images), size, image_path)


def merge_images(images, size):
    return inverse_transform(images)


def inverse_transform(images):
    return (images+1.)/2.


def to_json(output_path, *layers):
    with open(output_path, "w") as layer_f:
        lines = ""
        for w, b, bn in layers:
            layer_idx = w.name.split('/')[0].split('h')[1]
            B = b.eval()
            if "lin/" in w.name:
                W = w.eval()
                depth = W.shape[1]
            else:
                W = np.rollaxis(w.eval(), 2, 0)
                depth = W.shape[0]
            biases = {"sy": 1, "sx": 1, "depth": depth,
                      "w": ['%.2f' % elem for elem in list(B)]}
            if bn is not None:
                gamma = bn.gamma.eval()
                beta = bn.beta.eval()
                gamma = {"sy": 1, "sx": 1, "depth": depth,
                         "w": ['%.2f' % elem for elem in list(gamma)]}
                beta = {"sy": 1, "sx": 1, "depth": depth,
                        "w": ['%.2f' % elem for elem in list(beta)]}
            else:
                gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
                beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}
            if "lin/" in w.name:
                fs = []
                for w in W.T:
                    fs.append({"sy": 1, "sx": 1, "depth": W.shape[0],
                               "w": ['%.2f' % elem for elem in list(w)]})
                lines += """
                    var layer_%s = {
                        "layer_type": "fc",
                        "sy": 1, "sx": 1,
                        "out_sx": 1, "out_sy": 1,
                        "stride": 1, "pad": 0,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (
                        layer_idx.split('_')[0], W.shape[1], W.shape[0],
                        biases, gamma, beta, fs)
            else:
                fs = []
                for w_ in W:
                    fs.append({
                        "sy": 5, "sx": 5, "depth": W.shape[3],
                        "w": ['%.2f' % elem for elem in list(w_.flatten())]})
                lines += """
                    var layer_%s = {
                        "layer_type": "deconv",
                        "sy": 5, "sx": 5,
                        "out_sx": %s, "out_sy": %s,
                        "stride": 2, "pad": 1,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (
                        layer_idx, 2 ** (int(layer_idx) + 2),
                        2 ** (int(layer_idx) + 2),
                        W.shape[0], W.shape[3], biases, gamma, beta, fs)
        layer_f.write(" ".join(lines.replace("'", "").split()))


def make_gif(images, fname, duration=2, true_image=False):
    import moviepy.editor as mpy

    def make_frame(t):
        try:
            x = images[int(len(images) / duration * t)]
        except:
            x = images[-1]
        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x + 1) / 2 * 255).astype(np.uint8)
    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.write_gif(fname, fps=len(images) / duration)


def visualize(sess, dcgan, config, option):
    if option == 0:
        z_sample = np.random.uniform(
            -0.5, 0.5, size=(config.batch_size, dcgan.z_dim))
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
        save_images(
            samples, [8, 8],
            './samples/test_%s.png' % strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    elif option == 1:
        values = np.arange(0, 1, 1.0 / config.batch_size)
        for idx in xrange(100):
            print(" [*] %d" % idx)
            z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]
            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
            save_images(
                samples, [8, 8], './samples/test_arange_%s.png' % (idx))
    elif option == 2:
        values = np.arange(0, 1, 1.0 / config.batch_size)
        for idx in [random.randint(0, 99) for _ in xrange(100)]:
            print(" [*] %d" % idx)
            z = np.random.uniform(-0.2, 0.2, size=(dcgan.z_dim))
            z_sample = np.tile(z, (config.batch_size, 1))
            # z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]
            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
            make_gif(samples, './samples/test_gif_%s.gif' % (idx))
    elif option == 3:
        values = np.arange(0, 1, 1.0 / config.batch_size)
        for idx in xrange(100):
            print(" [*] %d" % idx)
            z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]
            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
            make_gif(samples, './samples/test_gif_%s.gif' % (idx))
    elif option == 4:
        image_set = []
        values = np.arange(0, 1, 1.0 / config.batch_size)
        for idx in xrange(100):
            print(" [*] %d" % idx)
            z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]
            image_set.append(
                sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))
            make_gif(image_set[-1], './samples/test_gif_%s.gif' % (idx))
        new_image_set = [
            merge(np.array([images[idx] for images in image_set]), [10, 10])
            for idx in range(64) + range(63, -1, -1)]
        make_gif(new_image_set, './samples/test_gif_merged.gif', duration=8)


def find_files(directory, pattern):
    result = []
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                result.append(filename)
                if len(result) % 1000 == 0:
                    print len(result)
    return result


def img2patches(img_in, psize):
    h, w = img_in.shape[:2]
    assert h >= 2 * psize and w >= 2 * psize
    img = img_in.copy()
    if len(img.shape) == 2:
        img = np.dstack([img, img, img])
    x1 = (h - psize) / 2
    y1 = (w - psize) / 2
    x2 = h / 2 - psize
    y2 = w / 2 - psize
    patch1 = img[x1:x1 + psize, y1:y1 + psize]
    patch2 = img[x2:x2 + psize * 2, y2:y2 + psize * 2]
    patch1 = scipy.misc.imresize(patch1, [64, 64])
    patch2 = scipy.misc.imresize(patch2, [64, 64])
    patch1 = patch1.astype('float32') / 127.5 - 1.0
    patch2 = patch2.astype('float32') / 127.5 - 1.0
    return patch1, patch2


def img2noise(img_in, noise_ratio=0.5):
    """
    Takes an input image and returns a noisy version of the image and
    the original image.
    """
    h, w = img_in.shape[:2]
    img = img_in.copy()
    # Create noise mask
    num_replace = h * w * noise_ratio
    noise_mask = [0] * int(num_replace) + [1] * (h * w - int(num_replace))
    random.shuffle(noise_mask)
    noise_mask = np.array(noise_mask).reshape(h, w)
    noise_mask = np.dstack([noise_mask, noise_mask, noise_mask])
    # Create noise array
    noise_array = np.random.randint(low=0, high=256, size=img.shape)
    # Zero out some elements of array and add noise there instead
    img = img * noise_mask + np.logical_not(noise_mask) * noise_array
    # Normalize to between 0-1
    patch1 = img.astype('float32') / 127.5 - 1.0
    patch2 = img_in.astype('float32') / 127.5 - 1.0
    return patch1, patch2


def img2downsampled(img_in, down_sample_level, smooth=False):
    """
    When smooth is false does:
    Returns a version of the image that had been downsampled
    "down_sample_level" times and then upsampled back to the original size
    E.g. when down_smaple_level = 1, the image will be divided into blocks
    each with 4 pixels in them, and those values will be replaced with
    the average. When smooth is true, the image is upsampled with
    bilinear interpolation, so your output is blurred instead of blocky.
    This is written in a way that supports non-integer down_sample_level,
    but then there will be border effects.
    """
    img = img_in.copy()
    h, w = img_in.shape[:2]
    assert h >= 64 and w >= 64
    assert down_sample_level >= 0 and down_sample_level <= 7
    # assert img.dtype == np.uint8
    down_sample_size = int(np.floor(64 / 2 ** down_sample_level))
    patch2 = scipy.misc.imresize(img, [64, 64])
    if down_sample_level == 7:
        patch1 = np.zeros_like(patch2)
    else:
        interp = 'bilinear'
        if not smooth:
            interp = 'nearest'
        patch1 = scipy.misc.imresize(
            patch2, [down_sample_size, down_sample_size], interp=interp)
        patch1 = scipy.misc.imresize(patch1, [64, 64], interp=interp)
    patch1 = patch1.astype('float32') / 127.5 - 1.0
    patch2 = patch2.astype('float32') / 127.5 - 1.0
    return patch1, patch2
