import os
import shutil

# Define the paths
images_dir = 'images'
annotations_dir = 'annotations'
train_txt_path = 'train.txt'
test_txt_path = 'test.txt'
val_txt_path = 'val.txt'
train_output_dir = 'Train'
test_output_dir = 'Test'
val_output_dir = 'Val'

# Function to read the filenames from a text file
def read_filenames(file_path):
    with open(file_path, 'r') as file:
        filenames = [line.strip() for line in file]
    return filenames

# Function to copy files to the target directory
def copy_files(filenames, img_src_dir, ann_src_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for filename in filenames:
        img_src_path = os.path.join(img_src_dir, filename)
        ann_src_path = os.path.join(ann_src_dir, filename.replace('.jpg', '.xml'))
        dest_path = os.path.join(dest_dir, filename)
        if os.path.exists(img_src_path):
            shutil.copy(img_src_path, dest_path)
        else:
            print(f"Warning: {img_src_path} does not exist and will be skipped.")
        if os.path.exists(ann_src_path):
            shutil.copy(ann_src_path, dest_path)
        else:
            print(f"Warning: {ann_src_path} does not exist and will be skipped.")

# Read the train and test filenames
train_filenames = read_filenames(train_txt_path)
test_filenames = read_filenames(test_txt_path)
val_filenames = read_filenames(val_txt_path)

# Copy the train images
copy_files(train_filenames, images_dir, annotations_dir, train_output_dir)

# Copy the test images
copy_files(test_filenames, images_dir, annotations_dir, test_output_dir)

copy_files(val_filenames, images_dir, annotations_dir, val_output_dir)

print("Files have been successfully copied to the Train and Test directories.")