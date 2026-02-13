from ultralytics import YOLO
import cv2, requests, time
from datetime import datetime

FIREBASE_URL = "https://countbottle-default-rtdb.firebaseio.com"

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

last_firebase_update = 0


def send_to_firebase(detected):
    try:
        data = {
            "bottle_detected": 1 if detected else 0,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        response = requests.put(
            f"{FIREBASE_URL}/bottle_detection.json",
            json=data
        )

        print("Sent:", data)
        print("Response:", response.status_code)

    except Exception as e:
        print("Firebase exception:", e)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    r = model(frame)[0]

    # Bottle class id = 39
    bottle_count = sum(int(b.cls[0]) == 39 and b.conf[0] > 0.7 for b in r.boxes)

    # Convert to 1 or 0
    bottle_detected = bottle_count > 0

    frame = r.plot()
    cv2.putText(frame, f"Bottles: {bottle_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Send every 2 seconds
    if time.time() - last_firebase_update >= 2:
        send_to_firebase(bottle_detected)
        last_firebase_update = time.time()

    cv2.imshow("Bottle Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
