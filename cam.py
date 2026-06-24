import cv2
import threading
import time
import queue
import numpy as np
import tensorflow as tf
from deepface import DeepFace
from ultralytics import YOLO

model = tf.keras.models.load_model("efficient_gender_model.keras")
yolo_model = YOLO('yolov8n.pt')

classes = ['man', 'woman']
global_registry = {}
analyzed_ids = set()
analysis_queue = queue.Queue(maxsize=20)

class RTSPStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.running = True
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            status, frame = self.cap.read()
            if status:
                self.frame = frame

    def stop(self):
        self.running = False
        self.cap.release()

def classification_worker():
    while True:
        track_id, crop, h, h_frame = analysis_queue.get()
        if track_id is None:
            break
        start_time=time.time()
        try:
            if h > h_frame * 0.3:
                try:
                    face_objs = DeepFace.extract_faces(
                        img_path=crop,
                        detector_backend='retinaface',
                        enforce_detection=True,
                        align=False
                    )
                    area = face_objs[0]['facial_area']
                    fx, fy, fw, fh = area['x'], area['y'], area['w'], area['h']
                    target_crop = crop[fy:fy+fh, fx:fx+fw]
                except ValueError:
                    target_crop = crop[0:int(h*0.4), :]
            else:
                target_crop = crop

            if target_crop.size > 0:
                target_crop = cv2.resize(target_crop, (96, 96))
                if len(target_crop.shape) == 2:
                    target_crop = cv2.cvtColor(target_crop, cv2.COLOR_GRAY2RGB)
                elif target_crop.shape[2] == 4:
                    target_crop = cv2.cvtColor(target_crop, cv2.COLOR_RGBA2RGB)
                else:
                    target_crop = cv2.cvtColor(target_crop, cv2.COLOR_BGR2RGB)

                target_input = target_crop.astype(np.float32) / 255.0
                target_input = np.expand_dims(target_input, axis=0)

                predictions = model(target_input, training=False).numpy()[0]
               
                idx = np.argmax(predictions)
                confidence = predictions[idx] * 100
                
               	end_time=time.time()
               	inference_time=( end_time - start_time ) * 1000
               	
                global_registry[track_id] = (classes[idx], confidence, inference_time)
               
        except Exception:
            global_registry[track_id] = ("error", 0.0, 0.0)
           
        analysis_queue.task_done()

threading.Thread(target=classification_worker, daemon=True).start()

vs = RTSPStream('rtsp://admin:admin123@10.101.0.17:554/avstream/channel=3/stream=1.sdp')

while True:
    frame = vs.frame
    if frame is None:
        continue

    h_frame, w_frame, _ = frame.shape
    yolo_start = time.time()
    results = yolo_model.track(frame, classes=[0], conf=0.4, persist=True, tracker="bytetrack.yaml", verbose=False)
    yolo_time = (time.time() - yolo_start ) * 1000
    active_instances = 0
	
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        active_instances=len(track_ids)

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1

            if w < 20 or h < 50:
                continue

            if track_id in global_registry:
                label_text, confidence, inf_time = global_registry[track_id]
               
                if label_text == "error":
                    color = (0, 0, 255)
                    label = f"ID:{track_id} AI Error"
                elif confidence < 70.0:
                    color = (128, 0, 255)
                    label = f"ID:{track_id} {label_text} {confidence:.1f}% ({inf_time:.0f}ms)"
                else:
                    color = (0, 255, 0) if label_text == 'man' else (0, 0, 255)
                    label = f"ID:{track_id} {label_text} {confidence:.1f}% ({inf_time:.0f}ms)"
                   
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
           
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, f"ID:{track_id} Analyzing...", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
               
                if track_id not in analyzed_ids and not analysis_queue.full():
                    crop = frame[y1:y2, x1:x2].copy()
                    analyzed_ids.add(track_id)
                    analysis_queue.put((track_id, crop, h, h_frame))
                    
    cv2.putText(frame, f"Active instances: {active_instances}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Yolo inference: {yolo_time:.0f}ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("MEC Optimized Surveillance", frame)
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

analysis_queue.put((None, None, None, None))
vs.stop()
cv2.destroyAllWindows()
