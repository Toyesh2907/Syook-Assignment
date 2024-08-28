import argparse
import os
import cv2
import numpy as np
from ultralytics import YOLO

def load_model(weights_path):
    model = YOLO(weights_path)
    return model

def perform_inference(model, image):
    results = model(image)
    return results

def save_cropped_person_images(image, person_boxes, output_dir, base_filename, padding=5):
    os.makedirs(output_dir, exist_ok=True)
    cropped_images_paths = []
    
    for i, box in enumerate(person_boxes):
        x1, y1, x2, y2 = map(int, box)
        
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        y2 = min(image.shape[0], y2 + padding)
        
        cropped_person_image = image[y1:y2, x1:x2]
        
        
        cropped_image_path = os.path.join(output_dir, f"{base_filename}_person_{i}.jpg")
        cv2.imwrite(cropped_image_path, cropped_person_image)
        cropped_images_paths.append(cropped_image_path)
    
    return cropped_images_paths

def draw_bounding_boxes(image, boxes, labels, confidences):
    
    colors = [
        (0, 255, 0),    
        (0, 0, 255),    
        (255, 0, 0),   
        (255, 255, 0), 
        (0, 255, 255),  
        (255, 0, 255),  
        (128, 128, 0),  
        (128, 0, 128), 
        (0, 128, 128), 
        (255, 128, 0)   
    ]
    j=0
    for box, label, confidence in zip(boxes, labels, confidences):
        if(j>len(colors)-1): j=0
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), colors[j], 1)
        cv2.putText(image, f"{label}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[j], 1)
        j+=1;
    return image

def process_cropped_images(input_dir, output_dir, ppe_det_model):
    os.makedirs(output_dir, exist_ok=True)
    
    for image_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_name)
        image = cv2.imread(image_path)
        
        
        ppe_results = perform_inference(ppe_det_model, image)
        
        if len(ppe_results) > 0 and len(ppe_results[0].boxes) > 0:
            
            ppe_boxes = ppe_results[0].boxes.xyxy.cpu().numpy()
            ppe_labels = [ppe_det_model.names[int(cls)] for cls in ppe_results[0].boxes.cls.cpu().numpy()]
            ppe_confidences = ppe_results[0].boxes.conf.cpu().numpy()
            
            image_with_ppe = draw_bounding_boxes(image, ppe_boxes, ppe_labels, ppe_confidences)
            
            output_image_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_name))[0]}_ppe.jpg")
            cv2.imwrite(output_image_path, image_with_ppe)

def main(input_dir, output_dir, person_det_model_path, ppe_det_model_path, output_dir_cropped_images):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_cropped_images, exist_ok=True)

    person_det_model = load_model(person_det_model_path)
    ppe_det_model = load_model(ppe_det_model_path)

    # Person detection and cropping
    for image_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_name)
        image = cv2.imread(image_path)
        
        # Perform person detection
        person_results = perform_inference(person_det_model, image)
        
        if len(person_results) > 0 and len(person_results[0].boxes) > 0:
            # Get person detection results
            person_boxes = person_results[0].boxes.xyxy.cpu().numpy()
            
            # Save cropped person images with padding
            save_cropped_person_images(image, person_boxes, output_dir, os.path.splitext(image_name)[0], padding=5)

    # PPE detection on cropped images
    process_cropped_images(output_dir, output_dir_cropped_images, ppe_det_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Inference Script")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save cropped images')
    parser.add_argument('--person_det_model', type=str, required=True, help='Path to person detection model weights')
    parser.add_argument('--ppe_det_model', type=str, required=True, help='Path to PPE detection model weights')
    parser.add_argument('--output_dir_cropped_images', type=str, required=True, help='Directory to save output images with PPE annotations')
    
    args = parser.parse_args()
    
    main(args.input_dir, args.output_dir, args.person_det_model, args.ppe_det_model, args.output_dir_cropped_images)
