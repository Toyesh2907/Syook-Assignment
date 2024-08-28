import os

class_names = [
    "person", "hard-hat", "gloves", "mask", "glasses", "boots",
    "vest", "ppe-suit", "ear-protector", "safety-harness"
]

classes_to_filter = [
    "hard-hat", "gloves", "glasses", "boots",
    "vest", "ppe-suit", "ear-protector"
]

# mapping original to new classes
original_to_new_class_index = {class_names.index(cls): i for i, cls in enumerate(classes_to_filter)}

def filter_annotations(input_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            input_file_path = os.path.join(input_dir, filename)
            output_file_path = os.path.join(output_dir, filename)

            #  input file
            with open(input_file_path, 'r') as input_file:
                lines = input_file.readlines()

            # Filter and remap class indices
            filtered_lines = []
            for line in lines:
                parts = line.split()
                class_id = int(parts[0])
                if class_id in original_to_new_class_index:
                    new_class_id = original_to_new_class_index[class_id]
                    parts[0] = str(new_class_id)
                    filtered_lines.append(" ".join(parts) + "\n")

            # filtered output file
            with open(output_file_path, 'w') as output_file:
                output_file.writelines(filtered_lines)

            print(f"Processed {filename} - filtered annotations written to {output_file_path}")
# input annotation directory
input_directory = "D:/Machine Learning/Datasets/Syook_Dataset/datasets/datasets/output_directory_yolo"

# filtered annotation directory
output_directory = "D:/Machine Learning/Datasets/Syook_Dataset/datasets/datasets/output_directory_PPE"

filter_annotations(input_directory, output_directory)
