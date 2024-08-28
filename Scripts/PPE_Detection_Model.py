from ultralytics import YOLO

model_PPE = YOLO('yolov8n.pt').load('yolov8n-cls.pt')

model_PPE.train(data="D:/Machine Learning/Datasets/Syook_Dataset/datasets/datasets/yaml files/data_PPE.yaml", epochs=100, imgsz=640,batch=8,patience=5)



