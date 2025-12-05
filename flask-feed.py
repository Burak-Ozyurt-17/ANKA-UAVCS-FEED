import cv2
import math
import random
import time
import numpy as np
import requests
from ultralytics import YOLO
from flask import Flask, Response, jsonify
from flask_cors import CORS
from threading import Thread, Lock
from inputs import get_gamepad, UnpluggedError, devices

# --- Setup ---
PI_STREAM_URL = "http://192.168.1.10:8080/stream"
model = YOLO("best.pt")  # YOLO detection
app = Flask(__name__)
CORS(app)

# --- Shared state + locks ---
lock = Lock()
altitude = 50.0
temp = 25.0
humidity = 50.0
smoke = 55.0
fire = False
lat = 28.9784
lon = 41.0082
angle = 0.0
cam = True

controller_state = {
    "ABS_X": 0,
    "ABS_Y": 0,
    "ABS_Z": 0,
    "ABS_RX": 0,
    "BTN_THUMB": 0,
}

vx, vy = 0.0, 0.0
last_debug_time = 0.0

# --- Shared frame buffer ---
latest_frame = None
frame_lock = Lock()

# -----------------------------
# Pi stream reader thread
# -----------------------------
def capture_loop():
    global latest_frame
    while True:
        try:
            r = requests.get(PI_STREAM_URL, stream=True, timeout=5)
            data = b""
            for chunk in r.iter_content(chunk_size=4096):
                data += chunk
                while b'\xff\xd9' in data:  # JPEG end marker
                    idx = data.find(b'\xff\xd9') + 2
                    jpeg_frame = data[:idx]
                    data = data[idx:]
                    frame = cv2.imdecode(np.frombuffer(jpeg_frame, np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        with frame_lock:
                            latest_frame = frame
        except Exception as e:
            print(f"[WARN] Pi stream error: {e}")
            time.sleep(2)  # retry after short delay

# -----------------------------
# MJPEG generator for Flask
# -----------------------------
def generate_mjpeg():
    boundary = b"--frame\r\n"
    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is None:
            time.sleep(0.01)
            continue

        # --- YOLO detection ---
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
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                    cv2.circle(frame, (cx,cy), 4, (0,0,255), -1)
                    cv2.putText(frame, f"{class_name} ({cx},{cy})", (int(x1), int(y1)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        global fire
        fire = fire_detected

        # Draw center marker
        h, w, _ = frame.shape
        cv2.drawMarker(frame, (w//2, h//2), (0,0,255),
                       markerType=cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=2)

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        time.sleep(0.01)  # optional small delay

# -----------------------------
# Controller helper functions
# -----------------------------
def wait_for_controller():
    while not devices.gamepads:
        print("[INFO] Waiting for controller connection...")
        time.sleep(2)
    print(f"[INFO] Controller detected: {devices.gamepads[0].name}")

def read_controller_loop():
    global controller_state
    wait_for_controller()
    print("[INFO] Gamepad connection watcher started.")
    while True:
        try:
            events = get_gamepad()
            with lock:
                controller_state["connected"] = True
                for e in events:
                    if e.code in controller_state and controller_state[e.code] != e.state:
                        controller_state[e.code] = e.state
        except UnpluggedError:
            with lock:
                for k in controller_state:
                    controller_state[k] = 0
                controller_state["connected"] = False
            print("[WARN] Gamepad disconnected. Retrying connection...")
            time.sleep(1.0)
        except Exception as e:
            with lock:
                controller_state["connected"] = False
            print(f"[ERROR] Controller loop exception: {e}")
            time.sleep(2.0)

# -----------------------------
# Simulation / drone data
# -----------------------------
def updated_data():
    global altitude, lat, lon, angle, vx, vy, last_debug_time, cam
    SPEED_FACTOR = 0.000002
    ROTATION_SPEED = 2.0
    ALTITUDE_SPEED = 0.1
    ROLL_STRAFE_FACTOR = 0.6
    DEADZONE = 0.05
    SMOOTHING = 0.90
    ANGLE_SMOOTHING = 0.85
    LOOP_DELAY = 0.04  # 25Hz

    angular_velocity = 0.0

    def normalize90(value):
        return max(-1.0, min(1.0, value / 90.0))

    while True:
        with lock:
            snapshot = controller_state.copy()

        cam = snapshot.get("BTN_THUMB", 0) == 0

        roll = -normalize90(snapshot.get("ABS_X", 0))
        pitch = normalize90(snapshot.get("ABS_Y", 0))
        throttle = normalize90(snapshot.get("ABS_Z", 0))
        yaw = normalize90(snapshot.get("ABS_RX", 0))

        roll = 0.0 if abs(roll) < DEADZONE else roll
        pitch = 0.0 if abs(pitch) < DEADZONE else pitch
        throttle = 0.0 if abs(throttle) < DEADZONE else throttle
        yaw = 0.0 if abs(yaw) < DEADZONE else yaw

        angular_velocity = angular_velocity * ANGLE_SMOOTHING + yaw * (1 - ANGLE_SMOOTHING)
        angle = (angle + angular_velocity * ROTATION_SPEED) % 360.0

        forward = -pitch
        strafe = roll * ROLL_STRAFE_FACTOR
        if forward != 0.0 or strafe != 0.0:
            rad = math.radians(angle)
            dx = (math.cos(rad) * forward - math.sin(rad) * strafe) * SPEED_FACTOR
            dy = (math.sin(rad) * forward + math.cos(rad) * strafe) * SPEED_FACTOR
            vx = vx * SMOOTHING + dx * (1 - SMOOTHING)
            vy = vy * SMOOTHING + dy * (1 - SMOOTHING)
        else:
            vx *= SMOOTHING
            vy *= SMOOTHING

        with lock:
            lat += vx
            lon += vy
            altitude = max(0.0, min(120.0, altitude + throttle * ALTITUDE_SPEED))

        now = time.time()
        if now - last_debug_time > 1.0:
            print(f"[DEBUG] roll={roll:.2f} pitch={pitch:.2f} yaw={yaw:.2f} throttle={throttle:.2f}")
            print(f"        angle={angle:.2f}, vx={vx:.8f}, vy={vy:.8f}, alt={altitude:.2f}")
            last_debug_time = now

        time.sleep(LOOP_DELAY)

# -----------------------------
# Flask routes
# -----------------------------
@app.route("/data")
def data():
    global altitude, temp, humidity, smoke, fire, lat, lon
    with lock:
        temp += random.uniform(-0.08, 0.08)
        humidity += random.uniform(-0.1, 0.1)
        smoke += random.uniform(-0.3, 0.3)
        return jsonify({
            "altitude": round(altitude, 2),
            "temp": round(temp, 2),
            "humidity": round(humidity, 2),
            "smoke": round(smoke, 2),
            "fire": fire,
            "latitude": round(lat, 7),
            "longitude": round(lon, 7),
            "angle": round(angle, 2),
            "controller_info": [getattr(gp, "name", "Unknown") for gp in devices.gamepads]
        })

@app.route("/debug")
def debug():
    with lock:
        normalized = {k: round(v / 90.0, 3) for k, v in controller_state.items() if k != "connected"}
        return jsonify({
            "controller_raw": controller_state,
            "normalized": normalized,
            "velocity": {"vx": vx, "vy": vy},
            "position": {"lat": lat, "lon": lon, "altitude": altitude, "angle": angle}
        })

@app.route("/")
def stream():
    return Response(generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    Thread(target=capture_loop, daemon=True).start()
    Thread(target=read_controller_loop, daemon=True).start()
    Thread(target=updated_data, daemon=True).start()
    app.run(host="0.0.0.0", port=8000, threaded=True)
