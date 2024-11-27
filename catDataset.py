from ultralytics import YOLO

model = YOLO('yolo11s.pt')  

results = model.train(data='datasets/data.yaml', epochs=50, batch=16, imgsz=640, name='modelo_gatos') 

results = model.val()

