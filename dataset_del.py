import os

images_path =r"D:\steel\raw_dataset\452_images\dataset_span"
labels_path = r"C:\Users\STUDENT\Downloads\457_span\labels\train"   # Example: r"P:\steel\labels"

# Get all label names (without .txt)
label_files = {os.path.splitext(f)[0] for f in os.listdir(labels_path) if f.endswith('.txt')}

# Go through image files
for image_file in os.listdir(images_path):
    if image_file.endswith(('.jpg', '.png')):
        base_name = os.path.splitext(image_file)[0]
        if base_name not in label_files:
            image_full_path = os.path.join(images_path, image_file)
            os.remove(image_full_path)
            print(f"Deleted image with no label: {image_file}")
