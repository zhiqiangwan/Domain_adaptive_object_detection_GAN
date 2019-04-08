"""
Find the image ids that have the desired object. Filter out the other image ids.
"""

from bs4 import BeautifulSoup
import os

annotation_dir = '../../datasets/VOCdevkit/VOC2012/Annotations'
image_set_filename_before_filter = '../../datasets/VOCdevkit/VOC2012/ImageSets/Main/val.txt'
image_set_filename_after_filter = '../../datasets/VOCdevkit/VOC2012_CAR/ImageSets/Main'
txt_filename = os.path.join(image_set_filename_after_filter, 'test.txt')

if not os.path.exists(image_set_filename_after_filter):
    os.makedirs(image_set_filename_after_filter)

with open(image_set_filename_before_filter, 'r') as f:
    image_ids = [line.strip() for line in f]

image_filenames = []
for image_id in image_ids:

    with open(os.path.join(annotation_dir, image_id + '.xml')) as f:
        soup = BeautifulSoup(f, 'xml')

    # In case we want to return the folder in addition to the image file name. Relevant for determining which dataset
    # an image belongs to.
    folder = soup.folder.text
    objects = soup.find_all('object')  # Get a list of all objects in this image.

    # Parse the data for each object.
    for obj in objects:
        class_name = obj.find('name', recursive=False).text
        if class_name == 'car':
            image_filenames.append(image_id)
            break

with open(txt_filename, 'w') as f:
    for item in image_filenames:
        f.write("%s\n" % item)



