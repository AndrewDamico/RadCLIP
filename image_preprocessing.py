#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

# pip3 install opencv-python

preprocess_dict = {
    "dicom_id": "",
    "subject_id": "",
    "study_id": "",
    "orig_img_path": "",
    "orig_img_height": 0,
    "orig_img_width": 0,
    "resized_file_name": "",
    "crop_img_height": 0,
    "crop_img_width": 0,
    "padding_delta_height": 0,
    "padding_delta_width": 0,
    "padded_img_height": 0,
    "padded_img_width": 0,
    "resized_img_height": 0,
    "resized_img_width": 0,
}

BASE_DIR = '/Volumes/PRO-G40/msds498'
output_path = f'{BASE_DIR}/output/all'
population_metadata = f'{output_path}/population_images_labels.csv'


def image_resize_padding(img_df, img_size=224):
    images = []
    for index, row in img_df.iterrows():
        img_path = row['image']
        img_height = row['rows']
        img_width = row['columns']
        print(f'Path={img_path}')
        print(f'Image Size = {img_height} x {img_width}')

        min_max_pixel_vals = Image.open(img_path).getextrema()
        print(f'Original Image minimum and maximum pixel values = {min_max_pixel_vals}')
        # height = width
        if img_height == img_width:
            resized_shape = (img_size, img_size)
            offset = (0, 0)

        # height > width
        elif img_height > img_width:
            resized_shape = (img_size, round(img_size * img_width / img_height))
            offset = (0, (img_size - resized_shape[1]) // 2)

        else:
            resized_shape = (round(img_size * img_height / img_width), img_size)
            offset = ((img_size - resized_shape[0]) // 2, 0)

        resized_shape = (resized_shape[1], resized_shape[0])
        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, resized_shape).astype(np.uint8)
        print(f'Resized Image Size = {img_resized.shape[0]} x {img_resized.shape[1]}')

        delta_height = img_size - img_resized.shape[0]
        delta_width = img_size - img_resized.shape[1]
        print(f'delta_height={delta_height}; delta_width={delta_width}')
        pad_width = delta_width // 2
        pad_height = delta_height // 2
        padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
        pil_image = Image.fromarray(img_resized)
        padded_img = ImageOps.expand(pil_image, padding)
        images.append(padded_img)
    return images


def crop_image(img_path):
    import cv2
    import numpy as np

    # load image as grayscale
    img = cv2.imread(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    hh, ww = thresh.shape

    # make bottom 2 rows black where they are white the full width of the image
    thresh[hh - 3:hh, 0:ww] = 0

    # get bounds of white pixels
    white = np.where(thresh == 255)
    xmin, ymin, xmax, ymax = np.min(white[1]), np.min(white[0]), np.max(white[1]), np.max(white[0])
    print(xmin, xmax, ymin, ymax)

    # crop the image at the bounds adding back the two blackened rows at the bottom
    crop = img[ymin:ymax + 3, xmin:xmax]

    crop_pil = Image.fromarray(crop)

    # save resulting masked image
    #     cv2.imwrite(f'{cropped_path}/xray_chest_thresh.jpg', thresh)

    # display result
    #     cv2.imshow("thresh", thresh)
    #     cv2.imshow("crop", crop)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    return crop_pil


def image_padding(img_path, img_height, img_width):
    if img_height > img_width:
        delta_height = 0
        delta_width = img_height - img_width
    elif img_height < img_width:
        delta_height = img_width - img_height
        delta_width = 0
    else:
        delta_height = 0
        delta_width = 0

    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_height, pad_width, delta_height - pad_height, delta_width - pad_width)
    orig_img = Image.open(img_path)
    padded_img = ImageOps.expand(orig_img, padding)
    # min_max_pixel_vals = orig_img.getextrema()
    # print(f'Original Image minimum and maximum pixel values = {min_max_pixel_vals}')

    return padded_img, delta_height, delta_width


def image_resize(img, img_size=448):
    resize_tp = (img_size, img_size)
    resized_img = img.resize(resize_tp, Image.LANCZOS)
    return resized_img


def image_padding_resize(img_df, output_dir, metadata_filename, img_size=448):
    cropped_dir = f'{output_dir}/cropped'
    padded_dir = f'{output_dir}/padded'
    resized_dir = f'{output_dir}/resized'
    preprocess_metadata = []
    total_imgs = img_df.dicom_id.count()
    for index, row in img_df.iterrows():
        print(f'Processing image # {index}/{total_imgs}')
        img_path = row['image']
        metadata = {}
        metadata['dicom_id'] = row['dicom_id']
        metadata['subject_id'] = str(row['subject_id'])
        metadata['study_id'] = row['study_id']
        metadata['orig_img_path'] = row['image']
        metadata['orig_img_height'] = row['rows']
        metadata['orig_img_width'] = row['columns']  # metadata['']

        dicom_id = row['dicom_id']
        sub_id = str(row['subject_id'])
        st_id = row['study_id']
        file_name = f'{sub_id}_{st_id}_{dicom_id}.jpg'

        print(f'   - Cropping the image - {sub_id}_{st_id}_{dicom_id}')
        crop_img = crop_image(f'{BASE_DIR}/{img_path}')
        crop_img_size = crop_img.size
        crop_path = f'{cropped_dir}/{file_name}'
        crop_img.save(crop_path)
        metadata['resized_file_name'] = file_name
        metadata['crop_img_height'] = crop_img_size[0]
        metadata['crop_img_width'] = crop_img_size[1]

        print(f'   - Padding the image - {sub_id}_{st_id}_{dicom_id}')
        padded_img, delta_height, delta_width = image_padding(crop_path, crop_img_size[0], crop_img_size[1])
        padded_img_size = padded_img.size
        padded_path = f'{padded_dir}/{file_name}'
        padded_img.save(padded_path)
        metadata['padding_delta_height'] = delta_height
        metadata['padding_delta_width'] = delta_width
        metadata['padded_img_height'] = padded_img_size[0]
        metadata['padded_img_width'] = padded_img_size[1]

        print(f'   - Resizing the image - {sub_id}_{st_id}_{dicom_id} to 224x224 size')
        resized_img = image_resize(padded_img)
        resized_img_size = resized_img.size
        resized_path = f'{resized_dir}/{file_name}'
        resized_img.save(resized_path)
        metadata['resized_img_height'] = resized_img_size[0]
        metadata['resized_img_width'] = resized_img_size[1]
        preprocess_metadata.append(metadata)

    preprocess_meta_df = pd.DataFrame(preprocess_metadata)
    preprocess_meta_df.to_csv(f"{output_dir}/{metadata_filename}", index=False)

    return


def mean_std(imgs_loc):
    from pathlib import Path
    import cv2
    import numpy as np
    imageFilesDir = Path(imgs_loc)
    files = list(imageFilesDir.rglob('*.jpg'))

    mean = np.array([0., 0., 0.])
    stdTemp = np.array([0., 0., 0.])
    std = np.array([0., 0., 0.])

    numSamples = len(files)

    for i in range(numSamples):
        im = cv2.imread(str(files[i]))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im.astype(float) / 255.

        for j in range(3):
            mean[j] += np.mean(im[:, :, j])

    mean = (mean / numSamples)

    for i in range(numSamples):
        im = cv2.imread(str(files[i]))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im.astype(float) / 255.
        for j in range(3):
            stdTemp[j] += ((im[:, :, j] - mean[j]) ** 2).sum() / (im.shape[0] * im.shape[1])

    std = np.sqrt(stdTemp / numSamples)

    print(mean)
    print(std)

    return


def main():
    print("Preprocessing Images")
    population_df = pd.read_csv(population_metadata)
    image_padding_resize(population_df, output_path, "preprocess_metadata_448.csv")  #
    print("Preprocessing Images - done!")

    mean_std(f'{output_path}/resized')


if __name__ == "__main__":
    main()
