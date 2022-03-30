import os
from pathlib import Path

DATASET_ROOT = f"{os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))}/datasets/VaTeX/raw_videos/"
# TRAIN_ROOT = os.path.join(DATASET_ROOT, "train")
# VALID_ROOT = os.path.join(DATASET_ROOT, "val")
# TEST_ROOT = os.path.join(DATASET_ROOT, "test")
TRAIN_ROOT = DATASET_ROOT
VALID_ROOT = DATASET_ROOT
TEST_ROOT = DATASET_ROOT

TRAIN_FRAMES_ROOT = os.path.join(DATASET_ROOT, "train_frames")
VALID_FRAMES_ROOT = os.path.join(DATASET_ROOT, "val_frames")
TEST_FRAMES_ROOT = os.path.join(DATASET_ROOT, "test_frames")

TRAIN_SOUND_ROOT = os.path.join(DATASET_ROOT, "train_sound")
VALID_SOUND_ROOT = os.path.join(DATASET_ROOT, "val_sound")
TEST_SOUND_ROOT = os.path.join(DATASET_ROOT, "test_sound")

CATEGORIES_PATH = "resources/categories.json"
CLASSES_PATH = "resources/classes.json"

TRAIN_METADATA_PATH = "resources/700/train/kinetics_700_train.json"
# VAL_METADATA_PATH = "resources/700/val/kinetics_700_val.json"
VAL_METADATA_PATH = f"{os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))}/kinetics600/validate.json"
TEST_METADATA_PATH = "resources/700/test/kinetics_700_test.json"

SUB_CLASS_PATH = "resources/700/categories.json"
