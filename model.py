import os
import pyzed.sl as sl
import cv2
from ultralytics import YOLO
import torch

def detect_and_display(svo_file_path, model_path):
    
    print("Loading YOLO model...")
    model = YOLO(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_file_path)
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.MILLIMETER

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Error: Unable to open the SVO2 file.")
        return

    image_zed = sl.Mat()
    print("Starting real-time detection... Press 'q' to quit.")

    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            frame = image_zed.get_data()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            #frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)

            results = model.predict(frame_rgb, device=device, conf=0.1)  
            detections = results[0].boxes  

            for detection in detections:
                x1, y1, x2, y2 = map(int, detection.xyxy[0])
                confidence = detection.conf[0]
                class_id = int(detection.cls[0])

                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Hand: {confidence:.2f}"
                cv2.putText(frame_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            frame_display = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            cv2.imshow("YOLO Hand Detection", frame_display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("End of SVO file reached.")
            break

    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    svo_file_path = r"C:\Users\S4\Downloads\TEEP PROJECT\new\video\1119\HD720_SN22187596_14-13-42_flipped.svo2"  

    model_path = r"C:\Users\S4\Downloads\TEEP PROJECT\new\CUSTOM_TRAINED_YOLO_2.pt"  
    detect_and_display(svo_file_path, model_path)