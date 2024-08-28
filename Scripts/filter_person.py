import os

class_names = [
    "person", "hard-hat", "gloves", "mask", "glasses", "boots",
    "vest", "ppe-suit", "ear-protector", "safety-harness"
]

classes_to_filter = [
    "person"
]

# Get indices of the classes to filter
indices_to_filter = [class_names.index(cls) for cls in classes_to_filter]

def filter_annotations(input_dir, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            input_file_path = os.path.join(input_dir, filename)
            output_file_path = os.path.join(output_dir, filename)

            #input file
            with open(input_file_path, 'r') as input_file:
                lines = input_file.readlines()

            # Filter classes
            filtered_lines = [
                line for line in lines 
                if int(line.split()[0]) in indices_to_filter
            ]

            # filtered output file
            with open(output_file_path, 'w') as output_file:
                output_file.writelines(filtered_lines)

            print(f"Processed {filename} - filtered annotations written to {output_file_path}")

# input annotation directory
input_directory = "D:/Machine Learning/Datasets/Syook_Dataset/datasets/datasets/output_directory_yolo"

# filtered annotation directory
output_directory = "D:/Machine Learning/Datasets/Syook_Dataset/datasets/datasets/output_directory_persons"

filter_annotations(input_directory, output_directory)
