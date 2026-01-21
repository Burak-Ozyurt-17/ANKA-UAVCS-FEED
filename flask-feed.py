import cv2
import math, random, time
import numpy as np
import keyboard
import threading
from ultralytics import YOLO
from flask import Flask, Response, jsonify
from flask_cors import CORS

model = YOLO("best.pt")
cap = cv2.VideoCapture(1)
app = Flask(__name__)
CORS(app)

altitude = 50.0
temp = 25.0
humidity = 50.0
smoke = 55.0
fire = False
lat = 30.3306
lon = 40.7392
angle = 0.0
cam = True
SPEED_FACTOR = 0.000002


def wait_user_input():
    global lat, lon, altitude, angle
    while True:
        if keyboard.is_pressed("up"):
            altitude += 0.1
        if keyboard.is_pressed("down"):
            altitude -= 0.1

        radian_angle = angle * math.pi / 180

        if keyboard.is_pressed("w"):
            lon += math.sin(radian_angle) * SPEED_FACTOR
            lat += math.cos(radian_angle) * SPEED_FACTOR
        if keyboard.is_pressed("s"):
            lon += math.sin(radian_angle) * -SPEED_FACTOR
            lat += math.cos(radian_angle) * -SPEED_FACTOR
        if keyboard.is_pressed("a"):
            lon += math.sin(radian_angle + math.pi / 2) * SPEED_FACTOR
            lat += math.cos(radian_angle + math.pi / 2) * SPEED_FACTOR
        if keyboard.is_pressed("d"):
            lon += math.sin(radian_angle - math.pi / 2) * SPEED_FACTOR
            lat += math.cos(radian_angle - math.pi / 2) * SPEED_FACTOR

        time.sleep(0.02)


def generate_frames():
    global fire, cam
    h, w = 480, 640
    blank_frame = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(
        blank_frame,
        "Kamera Kapali",
        (50, h // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False)
        fire_detected = False
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                if class_name == "fire":
                    fire_detected = True
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    cv2.rectangle(
                        frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
                    )
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(
                        frame,
                        f"{class_name} ({cx},{cy})",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
        global fire
        if fire_detected == True:
            fire = True
        else:
            fire = False
        h, w, _ = frame.shape
        cv2.drawMarker(
            frame,
            (w // 2, h // 2),
            (0, 0, 255),
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=10,
            thickness=2,
        )  # MJPEG encode
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/data", methods=["GET", "POST"])
def data():
    global altitude, angle, temp, humidity, smoke, fire, lat, lon
    temp += random.uniform(-0.08, 0.08)
    humidity += random.uniform(-0.1, 0.1)
    smoke += random.uniform(-0.3, 0.3)
    return jsonify(
        {
            "altitude": round(altitude, 2),
            "temp": round(temp, 2),
            "humidity": round(humidity, 2),
            "smoke": round(smoke, 2),
            "fire": bool(fire),
            "latitude": round(lat, 7),
            "longitude": round(lon, 7),
            "angle": round(angle, 2),
        }
    )


@app.route("/camera", methods=["GET", "POST"])
def stream():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/", methods=["GET", "POST"])
def index():
    return "Sunucu Aktif"


if __name__ == "__main__":
    Input_Thread = threading.Thread(target=wait_user_input)
    Input_Thread.start()
    app.run(host="0.0.0.0", port=8000, debug=False)
