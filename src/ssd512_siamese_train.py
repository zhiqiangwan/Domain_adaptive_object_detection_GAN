import sys

sys.path.append('../')
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import glob

from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger, TensorBoard
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd512_Siamese import ssd_512
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation, SSDDataAugmentation_Siamese
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

img_height = 512  # Height of the model input images
img_width = 512  # Width of the model input images
img_channels = 3  # Number of color channels of the model input images
mean_color = [123, 117, 104]  # Per-channel mean of images. Do not change if use any of the pre-trained weights.
# The color channel order in the original SSD is BGR,
# so we'll have the model reverse the color channel order of the input images.
swap_channels = [2, 1, 0]
# The anchor box scaling factors used in the original SSD512 for the Pascal VOC datasets
# scales_pascal =
# The anchor box scaling factors used in the original SSD512 for the MS COCO datasets
scales_coco = [0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05]
scales = scales_coco
aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]]  # The anchor box aspect ratios used in the original SSD512; the order matters
two_boxes_for_ar1 = True
steps = [8, 16, 32, 64, 128, 256, 512]  # Space between two adjacent anchor box center points for each predictor layer.
# The offsets of the first anchor box center points from the top and left borders of the image
# as a fraction of the step size for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
clip_boxes = False  # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
# The variances by which the encoded target coordinates are divided as in the original implementation
variances = [0.1, 0.1, 0.2, 0.2]
normalize_coords = True
Model_Build = 'New_Model'  # 'Load_Model'
Optimizer_Type = 'SGD'  # 'Adam'  #
# Different batch_size will have different prediction loss.
batch_size = 8  # Change the batch size if you like, or if you run into GPU memory issues.
# alpha_distance = 0.0001  # Coefficient for the distance between the source and target feature maps.
G_loss_weights = [0.01, 0.01, 1.0]
D_loss_weights = [0.01, 0.01]

# 'City_to_foggy0_01_resize_600_1200' # 'City_to_foggy0_02_resize_600_1200'  # 'SIM10K_to_VOC07'
# 'SIM10K'  # 'Cityscapes_foggy_beta_0_01'  #  'City_to_foggy0_02_resize_400_800'
# 'SIM10K_to_VOC12_resize_400_800'
DatasetName = 'SIM10K_to_City_resize_400_800'  # 'City_to_foggy0_01_resize_400_800'  # 'SIM10K_to_VOC07_resize_400_800'  #
processed_dataset_path = './processed_dataset_h5/' + DatasetName
if not os.path.exists(processed_dataset_path):
    os.makedirs(processed_dataset_path)

checkpoint_path = '../trained_weights/SIM10K_to_City/current/G_weights0_01_D_weights_0_01'
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

csv_file_name = 'G_weights0_01_D_weights_0_01_training_log.csv'

if len(glob.glob(os.path.join(processed_dataset_path, '*.h5'))):
    Dataset_Build = 'Load_Dataset'
else:
    Dataset_Build = 'New_Dataset'

if DatasetName == 'SIM10K_to_VOC12_resize_400_800':
    resize_image_to = (400, 800)
    # The directories that contain the images.
    train_source_images_dir = '../../datasets/SIM10K/JPEGImages'
    train_target_images_dir = '../../datasets/VOCdevkit/VOC2012/JPEGImages'
    test_target_images_dir = '../../datasets/VOCdevkit/VOC2012/JPEGImages'

    # The directories that contain the annotations.
    train_annotation_dir = '../../datasets/SIM10K/Annotations'
    test_annotation_dir = '../../datasets/VOCdevkit/VOC2012/Annotations'

    # The paths to the image sets.
    train_source_image_set_filename = '../../datasets/SIM10K/ImageSets/Main/trainval10k.txt'
    # The trainset of VOC which has 'car' object is used as train_target.
    train_target_image_set_filename = '../../datasets/VOCdevkit/VOC2012_CAR/ImageSets/Main/train_target.txt'
    # The valset of VOC which has 'car' object is used as test.
    test_target_image_set_filename = '../../datasets/VOCdevkit/VOC2012_CAR/ImageSets/Main/test.txt'

    classes = ['background', 'car']  # Our model will produce predictions for these classes.
    train_classes = ['background', 'car', 'motorbike', 'person']  # The train_source dataset contains these classes.
    train_include_classes = [train_classes.index(one_class) for one_class in classes[1:]]
    # The test_target dataset contains these classes.
    val_classes = ['background', 'car',
                   'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'cat',
                   'chair', 'cow', 'diningtable', 'dog',
                   'horse', 'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor']
    val_include_classes = [val_classes.index(one_class) for one_class in classes[1:]]
    # Number of positive classes, 8 for domain Cityscapes, 20 for Pascal VOC, 80 for MS COCO, 1 for SIM10K
    n_classes = len(classes) - 1

elif DatasetName == 'SIM10K_to_VOC07_resize_400_800':
    resize_image_to = (400, 800)
    # The directories that contain the images.
    train_source_images_dir = '../../datasets/SIM10K/JPEGImages'
    train_target_images_dir = '../../datasets/VOCdevkit/VOC2007/JPEGImages'
    test_target_images_dir = '../../datasets/VOCdevkit/VOC2007/JPEGImages'

    # The directories that contain the annotations.
    train_annotation_dir = '../../datasets/SIM10K/Annotations'
    test_annotation_dir = '../../datasets/VOCdevkit/VOC2007/Annotations'

    # The paths to the image sets.
    train_source_image_set_filename = '../../datasets/SIM10K/ImageSets/Main/trainval10k.txt'
    # The trainset of VOC which has 'car' object is used as train_target.
    train_target_image_set_filename = '../../datasets/VOCdevkit/VOC2007_CAR/ImageSets/Main/train_target.txt'
    # The valset of VOC which has 'car' object is used as test.
    test_target_image_set_filename = '../../datasets/VOCdevkit/VOC2007_CAR/ImageSets/Main/test.txt'

    classes = ['background', 'car']  # Our model will produce predictions for these classes.
    train_classes = ['background', 'car', 'motorbike', 'person']  # The train_source dataset contains these classes.
    train_include_classes = [train_classes.index(one_class) for one_class in classes[1:]]
    # The test_target dataset contains these classes.
    val_classes = ['background', 'car',
                   'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'cat',
                   'chair', 'cow', 'diningtable', 'dog',
                   'horse', 'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor']
    val_include_classes = [val_classes.index(one_class) for one_class in classes[1:]]
    # Number of positive classes, 8 for domain Cityscapes, 20 for Pascal VOC, 80 for MS COCO, 1 for SIM10K
    n_classes = len(classes) - 1

elif DatasetName == 'SIM10K_to_City_resize_400_800':
    resize_image_to = (400, 800)
    # The directories that contain the images.
    train_source_images_dir = '../../datasets/SIM10K/JPEGImages'
    train_target_images_dir = '../../datasets/Cityscapes/JPEGImages'
    test_target_images_dir = '../../datasets/val_data_for_SIM10K_to_cityscapes/JPEGImages'

    # The directories that contain the annotations.
    train_annotation_dir = '../../datasets/SIM10K/Annotations'
    test_annotation_dir = '../../datasets/val_data_for_SIM10K_to_cityscapes/Annotations'

    # The paths to the image sets.
    train_source_image_set_filename = '../../datasets/SIM10K/ImageSets/Main/trainval10k.txt'
    train_target_image_set_filename = '../../datasets/Cityscapes/ImageSets/Main/train_source.txt'
    test_target_image_set_filename = '../../datasets/val_data_for_SIM10K_to_cityscapes/ImageSets/Main/test.txt'

    classes = ['background', 'car']  # Our model will produce predictions for these classes.
    train_classes = ['background', 'car', 'motorbike', 'person']  # The train_source dataset contains these classes.
    train_include_classes = [train_classes.index(one_class) for one_class in classes[1:]]
    # The test_target dataset contains these classes.
    val_classes = ['background', 'car']
    val_include_classes = 'all'
    # Number of positive classes, 8 for domain Cityscapes, 20 for Pascal VOC, 80 for MS COCO, 1 for SIM10K
    n_classes = len(classes) - 1

elif DatasetName == 'City_to_foggy0_02_resize_400_800':
    resize_image_to = (400, 800)
    # Introduction of PascalVOC: https://arleyzhang.github.io/articles/1dc20586/
    # The directories that contain the images.
    train_source_images_dir = '../../datasets/Cityscapes/JPEGImages'
    train_target_images_dir = '../../datasets/Cityscapes/JPEGImages'
    test_target_images_dir = '../../datasets/Cityscapes/JPEGImages'

    # The directories that contain the annotations.
    train_annotation_dir = '../../datasets/Cityscapes/Annotations'
    test_annotation_dir = '../../datasets/Cityscapes/Annotations'

    # The paths to the image sets.
    train_source_image_set_filename = '../../datasets/Cityscapes/ImageSets/Main/train_source.txt'
    train_target_image_set_filename = '../../datasets/Cityscapes/ImageSets/Main/train_target.txt'
    test_target_image_set_filename = '../../datasets/Cityscapes/ImageSets/Main/test.txt'
    # Our model will produce predictions for these classes.
    classes = ['background',
               'person', 'rider', 'car', 'truck',
               'bus', 'train', 'motorcycle', 'bicycle']
    train_classes = classes
    train_include_classes = 'all'
    val_classes = classes
    val_include_classes = 'all'
    # Number of positive classes, 8 for domain Cityscapes, 20 for Pascal VOC, 80 for MS COCO, 1 for SIM10K
    n_classes = len(classes) - 1

elif DatasetName == 'City_to_foggy0_01_resize_400_800':
    resize_image_to = (400, 800)
    # Introduction of PascalVOC: https://arleyzhang.github.io/articles/1dc20586/
    # The directories that contain the images.
    train_source_images_dir = '../../datasets/Cityscapes/JPEGImages'
    train_target_images_dir = '../../datasets/CITYSCAPES_beta_0_01/JPEGImages'
    test_target_images_dir = '../../datasets/CITYSCAPES_beta_0_01/JPEGImages'

    # The directories that contain the annotations.
    train_annotation_dir = '../../datasets/Cityscapes/Annotations'
    test_annotation_dir = '../../datasets/Cityscapes/Annotations'

    # The paths to the image sets.
    train_source_image_set_filename = '../../datasets/Cityscapes/ImageSets/Main/train_source.txt'
    train_target_image_set_filename = '../../datasets/Cityscapes/ImageSets/Main/train_target.txt'
    test_target_image_set_filename = '../../datasets/Cityscapes/ImageSets/Main/test.txt'
    # Our model will produce predictions for these classes.
    classes = ['background',
               'person', 'rider', 'car', 'truck',
               'bus', 'train', 'motorcycle', 'bicycle']
    train_classes = classes
    train_include_classes = 'all'
    val_classes = classes
    val_include_classes = 'all'
    # Number of positive classes, 8 for domain Cityscapes, 20 for Pascal VOC, 80 for MS COCO, 1 for SIM10K
    n_classes = len(classes) - 1

else:
    raise ValueError('Undefined dataset name.')


if Model_Build == 'New_Model':
    # 1: Build the Keras model.

    K.clear_session()  # Clear previous models from memory.

    # import tensorflow as tf
    # from keras.backend.tensorflow_backend import set_session
    #
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    # config.log_device_placement = True  # to log device placement (on which device the operation ran)
    # # (nothing gets printed in Jupyter, only if you run it standalone)
    # sess = tf.Session(config=config)
    # set_session(sess)  # set this TensorFlow session as the default session for Keras

    D_model, G_model = ssd_512(image_size=(img_height, img_width, img_channels),
                               n_classes=n_classes,
                               G_loss_weights=G_loss_weights,
                               D_loss_weights=D_loss_weights,
                               mode='training',
                               l2_regularization=0.0005,
                               scales=scales,
                               aspect_ratios_per_layer=aspect_ratios,
                               two_boxes_for_ar1=two_boxes_for_ar1,
                               steps=steps,
                               offsets=offsets,
                               clip_boxes=clip_boxes,
                               variances=variances,
                               normalize_coords=normalize_coords,
                               subtract_mean=mean_color,
                               swap_channels=swap_channels)

else:
    raise ValueError('Undefined Model_Build. Model_Build should be New_Model  or Load_Model')


if Dataset_Build == 'New_Dataset':
    # 1: Instantiate two `DataGenerator` objects: One for training, one for validation.

    # Optional: If you have enough memory, consider loading the images into memory for the reasons explained above.

    train_dataset = DataGenerator(dataset='train', load_images_into_memory=False, hdf5_dataset_path=None)
    val_dataset = DataGenerator(dataset='val', load_images_into_memory=False, hdf5_dataset_path=None)

    # 2: Parse the image and label lists for the training and validation datasets. This can take a while.
    # images_dirs, image_set_filenames, and annotations_dirs should have the same length
    train_dataset.parse_xml(images_dirs=[train_source_images_dir],
                            target_images_dirs=[train_target_images_dir],
                            image_set_filenames=[train_source_image_set_filename],
                            target_image_set_filenames=[train_target_image_set_filename],
                            annotations_dirs=[train_annotation_dir],
                            classes=train_classes,
                            include_classes=train_include_classes,
                            exclude_truncated=False,
                            exclude_difficult=False,
                            ret=False)

    val_dataset.parse_xml(images_dirs=[test_target_images_dir],
                          image_set_filenames=[test_target_image_set_filename],
                          annotations_dirs=[test_annotation_dir],
                          classes=val_classes,
                          include_classes=val_include_classes,
                          exclude_truncated=False,
                          exclude_difficult=True,
                          ret=False)

    # Optional: Convert the dataset into an HDF5 dataset. This will require more disk space, but will
    # speed up the training. Doing this is not relevant in case you activated the `load_images_into_memory`
    # option in the constructor, because in that cas the images are in memory already anyway. If you don't
    # want to create HDF5 datasets, comment out the subsequent two function calls.

    # After create these h5 files, if you have resized the input image, you need to reload these files. Otherwise,
    # the images and the labels will not change.

    train_dataset.create_hdf5_dataset(file_path=os.path.join(processed_dataset_path, 'dataset_train.h5'),
                                      resize=resize_image_to,
                                      variable_image_size=True,
                                      verbose=True)

    val_dataset.create_hdf5_dataset(file_path=os.path.join(processed_dataset_path, 'dataset_test.h5'),
                                    resize=False,
                                    variable_image_size=True,
                                    verbose=True)

    train_dataset = DataGenerator(dataset='train',
                                  load_images_into_memory=False,
                                  hdf5_dataset_path=os.path.join(processed_dataset_path, 'dataset_train.h5'),
                                  filenames=train_source_image_set_filename,
                                  target_filenames=train_target_image_set_filename,
                                  filenames_type='text',
                                  images_dir=train_source_images_dir,
                                  target_images_dir=train_target_images_dir)

    val_dataset = DataGenerator(dataset='val',
                                load_images_into_memory=False,
                                hdf5_dataset_path=os.path.join(processed_dataset_path, 'dataset_test.h5'),
                                filenames=test_target_image_set_filename,
                                filenames_type='text',
                                images_dir=test_target_images_dir)

elif Dataset_Build == 'Load_Dataset':
    # 1: Instantiate two `DataGenerator` objects: One for training, one for validation.

    # Load dataset from the created h5 file.
    train_dataset = DataGenerator(dataset='train',
                                  load_images_into_memory=False,
                                  hdf5_dataset_path=os.path.join(processed_dataset_path, 'dataset_train.h5'),
                                  filenames=train_source_image_set_filename,
                                  target_filenames=train_target_image_set_filename,
                                  filenames_type='text',
                                  images_dir=train_source_images_dir,
                                  target_images_dir=train_target_images_dir)

    val_dataset = DataGenerator(dataset='val',
                                load_images_into_memory=False,
                                hdf5_dataset_path=os.path.join(processed_dataset_path, 'dataset_test.h5'),
                                filenames=test_target_image_set_filename,
                                filenames_type='text',
                                images_dir=test_target_images_dir)

else:
    raise ValueError('Undefined Dataset_Build. Dataset_Build should be New_Dataset or Load_Dataset.')


# 4: Set the image transformations for pre-processing and data augmentation options.

# For the training generator:
ssd_data_augmentation = SSDDataAugmentation_Siamese(img_height=img_height,
                                                    img_width=img_width)

# For the validation generator:
convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=img_height, width=img_width)

# 5: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.

# The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
predictor_sizes = [G_model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
                   G_model.get_layer('fc7_mbox_conf').output_shape[1:3],
                   G_model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
                   G_model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                   G_model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                   G_model.get_layer('conv9_2_mbox_conf').output_shape[1:3],
                   G_model.get_layer('conv10_2_mbox_conf').output_shape[1:3]]

ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=scales,
                                    aspect_ratios_per_layer=aspect_ratios,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    clip_boxes=clip_boxes,
                                    variances=variances,
                                    matching_type='multi',
                                    pos_iou_threshold=0.5,
                                    neg_iou_limit=0.5,
                                    normalize_coords=normalize_coords)

# 6: Create the generator handles that will be passed to Keras' `fit_generator()` function.
# The input image and label are first processed by transformations. Then, the label will be further encoded by
# ssd_input_encoder. The encoded labels are classId and offset to each anchor box.
# G_train_generator = train_dataset.generate(batch_size=batch_size,
#                                            generator_type='G',
#                                            shuffle=True,
#                                            transformations=[ssd_data_augmentation],
#                                            label_encoder=ssd_input_encoder,
#                                            returns={'processed_images',
#                                                     'encoded_labels'},
#                                            keep_images_without_gt=False)

# D_train_generator = train_dataset.generate(batch_size=batch_size,
#                                            generator_type='D',
#                                            shuffle=True,
#                                            transformations=[ssd_data_augmentation],
#                                            label_encoder=ssd_input_encoder,
#                                            returns={'processed_images',
#                                                     'encoded_labels'},
#                                            keep_images_without_gt=False)

# val_generator = val_dataset.generate(batch_size=batch_size,
#                                      generator_type='G',
#                                      shuffle=False,
#                                      transformations=[convert_to_3_channels,
#                                                       resize],
#                                      label_encoder=ssd_input_encoder,
#                                      returns={'processed_images',
#                                               'encoded_labels'},
#                                      keep_images_without_gt=False)

# Get the number of samples in the training and validations datasets.
train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size = val_dataset.get_dataset_size()

print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))

# G_train_generator = train_dataset.generate(batch_size=batch_size,
#                                            generator_type='G',
#                                            shuffle=True,
#                                            transformations=[ssd_data_augmentation],
#                                            label_encoder=ssd_input_encoder,
#                                            returns={'processed_images',
#                                                     'encoded_labels'},
#                                            keep_images_without_gt=False)
#
# while True:
#     batch = next(G_train_generator)

def lr_schedule(epoch):
    if epoch < 20:
        return 0.0005
    elif epoch < 800:
        return 0.001
    elif epoch < 1000:
        return 0.0001
    else:
        return 0.00001

# Define model callbacks.
# TODO: Set the filepath under which you want to save the model.
model_checkpoint = ModelCheckpoint(filepath=os.path.join(checkpoint_path, 'epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5'),
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=False,
                                   save_weights_only=True,
                                   mode='auto',
                                   period=1)

# model_checkpoint.best to the best validation loss from the previous training
# model_checkpoint.best = 4.83704

csv_logger = CSVLogger(filename=os.path.join(checkpoint_path, csv_file_name),
                       separator=',',
                       append=True)

learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule,
                                                verbose=1)

terminate_on_nan = TerminateOnNaN()

TensorBoard_monitor = TensorBoard(log_dir=checkpoint_path)

callbacks = [model_checkpoint,
             csv_logger,
             learning_rate_scheduler,
             terminate_on_nan,
             TensorBoard_monitor]

callbacks_no_val = [learning_rate_scheduler,
                    terminate_on_nan]

# def lr_schedule_D(epoch):
#     return 0.0005


# learning_rate_scheduler_D = LearningRateScheduler(schedule=lr_schedule_D,
#                                                   verbose=1)

# callbacks_no_val_D = [learning_rate_scheduler_D,
#                       terminate_on_nan]


initial_epoch = 0
final_epoch = 1602
steps_per_G_epoch = 50
steps_per_D_epoch = 25

for initial_epoch in range(final_epoch):
    print('\n')
    print('Train generator.')
    G_train_generator = train_dataset.generate(batch_size=batch_size,
                                               generator_type='G',
                                               shuffle=True,
                                               transformations=[ssd_data_augmentation],
                                               label_encoder=ssd_input_encoder,
                                               returns={'processed_images',
                                                        'encoded_labels'},
                                               keep_images_without_gt=False)

    if initial_epoch % 20 == 19:
        val_generator = val_dataset.generate(batch_size=batch_size,
                                             generator_type='G',
                                             shuffle=False,
                                             transformations=[convert_to_3_channels,
                                                              resize],
                                             label_encoder=ssd_input_encoder,
                                             returns={'processed_images',
                                                      'encoded_labels'},
                                             keep_images_without_gt=False)

        history_G = G_model.fit_generator(generator=G_train_generator,
                                          steps_per_epoch=steps_per_G_epoch,
                                          epochs=initial_epoch+1,
                                          callbacks=callbacks,
                                          validation_data=val_generator,
                                          validation_steps=ceil(val_dataset_size/batch_size),
                                          initial_epoch=initial_epoch)

    else:
        history_G = G_model.fit_generator(generator=G_train_generator,
                                          steps_per_epoch=steps_per_G_epoch,
                                          epochs=initial_epoch+1,
                                          callbacks=callbacks_no_val,
                                          initial_epoch=initial_epoch)

    print('\n')
    print( 'Train discriminator.')
    D_train_generator = train_dataset.generate(batch_size=batch_size,
                                               generator_type='D',
                                               shuffle=True,
                                               transformations=[ssd_data_augmentation],
                                               label_encoder=ssd_input_encoder,
                                               returns={'processed_images',
                                                        'encoded_labels'},
                                               keep_images_without_gt=False)

    history_D = D_model.fit_generator(generator=D_train_generator,
                                      steps_per_epoch=steps_per_D_epoch,
                                      epochs=initial_epoch+1,
                                      callbacks=callbacks_no_val,
                                      initial_epoch=initial_epoch)


