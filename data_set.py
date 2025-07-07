import os
import random
import shutil

# Configuration
image_dir = r"D:\steel\real_time\images"
label_dir = r"D:\steel\real_time\labels"
output_dir = 'real_data_span'
split_ratio = 0.8  # 80% train, 20% val

# Create output directories
for split in ['train', 'val']:
    os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

# Supported image extensions (case-insensitive)
valid_exts = ['.jpg', '.jpeg', '.png']
images = [f for f in os.listdir(image_dir) if os.path.splitext(f)[1].lower() in valid_exts]

print(f"Total image files found: {len(images)}")

# Shuffle and split dataset
random.shuffle(images)
split_index = int(len(images) * split_ratio)
train_images = images[:split_index]
val_images = images[split_index:]

# Function to move image and label files
def move_files(image_list, split):
    for img_file in image_list:
        label_file = os.path.splitext(img_file)[0] + '.txt'

        img_src = os.path.join(image_dir, img_file)
        label_src = os.path.join(label_dir, label_file)

        img_dst = os.path.join(output_dir, 'images', split, img_file)
        label_dst = os.path.join(output_dir, 'labels', split, label_file)

        # Copy image
        shutil.copy(img_src, img_dst)

        # Copy label if it exists
        if os.path.exists(label_src):
            shutil.copy(label_src, label_dst)
        else:
            print(f"Warning: Missing label file for {img_file}")

# Move files to corresponding folders
move_files(train_images, 'train')
move_files(val_images, 'val')

# Summary
print(f"Train: {len(train_images)}, Val: {len(val_images)}")
