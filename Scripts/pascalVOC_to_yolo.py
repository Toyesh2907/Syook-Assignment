import os
import xml.etree.ElementTree as ET
from argparse import ArgumentParser

def convert_voc_to_yolo(xml_file, output_dir, classes):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    filename = os.path.splitext(os.path.basename(xml_file))[0]
    yolo_annotation = []

    for obj in root.iter('object'):
        difficult = obj.find('difficult')
        if difficult is not None and int(difficult.text) == 1:
            continue

        class_name = obj.find('name').text
        if class_name not in classes:
            continue

        class_id = classes.index(class_name)

        xmlbox = obj.find('bndbox')
        xmin = float(xmlbox.find('xmin').text)
        ymin = float(xmlbox.find('ymin').text)
        xmax = float(xmlbox.find('xmax').text)
        ymax = float(xmlbox.find('ymax').text)

        x_center = (xmin + xmax) / 2.0 / width
        y_center = (ymin + ymax) / 2.0 / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height

        yolo_annotation.append(f"{class_id} {x_center} {y_center} {box_width} {box_height}")

    yolo_output = os.path.join(output_dir, f"{filename}.txt")
    with open(yolo_output, 'w') as outfile:
        outfile.write("\n".join(yolo_annotation))

def main(input_dir, output_dir, classes_file):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(classes_file) as f:
        classes = f.read().strip().split()

    for xml_file in os.listdir(input_dir):
        if xml_file.endswith(".xml"):
            convert_voc_to_yolo(os.path.join(input_dir, xml_file), output_dir, classes)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input_dir", help="Directory containing Pascal VOC XML annotations")
    parser.add_argument("output_dir", help="Directory to save YOLO formatted annotations")
    parser.add_argument("classes_file", help="File containing class names (one per line)")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.classes_file)
