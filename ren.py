import os
import shutil

image_folder =r"D:\steel\raw_dataset\306_span_images\Copy of VID_20250605_114147"
labels_path = r"C:\Users\STUDENT\Downloads\306_span\labels\train"   
label_folder =  r"D:\steel\raw_dataset\306_span (2)\labels\train"    # Example: r"P:\steel\labels"

image_exts = [".jpg", ".jpeg", ".png"]

# Loop through image files
for filename in os.listdir(image_folder):
    name, ext = os.path.splitext(filename)

    if ext.lower() in image_exts:
        # Image rename
        old_image_path = os.path.join(image_folder, filename)
        new_image_name = name + "_set1" + ext
        new_image_path = os.path.join(image_folder, new_image_name)
        os.rename(old_image_path, new_image_path)

        # Label rename
        old_label_path = os.path.join(label_folder, name + ".txt")
        new_label_path = os.path.join(label_folder, name + "_set1.txt")
        if os.path.exists(old_label_path):
            os.rename(old_label_path, new_label_path)
        else:
            print(f"Label not found for: {filename}")
