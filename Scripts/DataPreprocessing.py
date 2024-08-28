#used for filtering datasets for person an PPE model training
import os 
import shutil
import random
def split_dataset(image_dir,label_dir,output_dir,train_ratio=0.8):
    train_image_dir = os.path.join(output_dir, 'images/train')
    val_image_dir = os.path.join(output_dir, 'images/val')
    train_label_dir = os.path.join(output_dir, 'labels/train')
    val_label_dir = os.path.join(output_dir, 'labels/val')
    
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(val_image_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)

    # Get list of images and corresponding labels
    images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    random.shuffle(images)

    # Calculate split index
    split_idx = int(len(images) * train_ratio)

    # Split the dataset
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    for img in train_images:
        label = img.replace('.jpg', '.txt')
        shutil.copy(os.path.join(image_dir, img), os.path.join(train_image_dir, img))
        shutil.copy(os.path.join(label_dir, label), os.path.join(train_label_dir, label))

    for img in val_images:
        label = img.replace('.jpg', '.txt')
        shutil.copy(os.path.join(image_dir, img), os.path.join(val_image_dir, img))
        shutil.copy(os.path.join(label_dir, label), os.path.join(val_label_dir, label))

    print(f"Training images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")

image_dir = "D:/Machine Learning/Datasets/Syook_Dataset/datasets/datasets/images"

label_dir_persons = "D:/Machine Learning//Datasets/Syook_Dataset/datasets/datasets/output_directory_persons"
output_dir_persons = "D:/Machine Learning/Datasets/Syook_Dataset/datasets/datasets/Train Test Data/person_data"


label_dir_PPE = "D:/Machine Learning//Datasets/Syook_Dataset/datasets/datasets/output_directory_PPE"
output_dir_PPE = "D:/Machine Learning/Datasets/Syook_Dataset/datasets/datasets/Train Test Data/PPE_data"

split_dataset(image_dir, label_dir_persons, output_dir_persons)
split_dataset(image_dir, label_dir_PPE, output_dir_PPE)
