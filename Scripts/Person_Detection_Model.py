from ultralytics import YOLO

model_person = YOLO('yolov8n.pt').load('yolov8n-cls.pt')

model_person.train(data="D:/Machine Learning/Datasets/Syook_Dataset/datasets/datasets/yaml files/data_person.yaml", epochs=100, imgsz=640,batch=8,patience=5)



