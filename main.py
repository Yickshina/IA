import cv2
from ultralytics import YOLO

model = YOLO('yolo11n.pt')

cap = cv2.VideoCapture(0)  

if not cap.isOpened():
    print("Error al abrir la cámara.")
    exit()

while(True):
    ret, frame = cap.read()

    if not ret:
        print("No se puede recibir el fotograma. Saliendo ...")
        break

    results = model.predict(source=frame)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = box.conf[0]
            cls = int(box.cls[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{model.names[cls]} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Detecciones en vivo', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()