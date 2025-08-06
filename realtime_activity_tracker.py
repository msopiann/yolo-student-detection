from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
from collections import defaultdict, deque, Counter
import time
import os
import pyautogui
import json
import pandas as pd

# === Configurations ===
activity_classes = ["hand-raising", "read", "write"]
activity_colors = {
    "hand-raising": (46, 184, 138),
    "read": (38, 98, 217),
    "write": (232, 140, 48),
    "inactive": (225, 54, 112),
}
class_thresholds = {"hand-raising": 0.1, "read": 0.1, "write": 0.1}
VOTING_WINDOW = 15
INACTIVE_DISPLAY_FRAMES = 60

# === Load Model & Tracker ===
model = YOLO(
    "D:/sopian/skripsi/code-for-model/student_behavior_analysis/yolov11m_best.pt"
)
tracker = DeepSort(max_age=30)

# === Open Webcam ===
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# === Fullscreen Display ===
screen_width, screen_height = pyautogui.size()
cv2.namedWindow("📹 Real-time Activity Tracking", cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    "📹 Real-time Activity Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
)

# === Save Output Video ===
os.makedirs("realtime_output", exist_ok=True)
output_path = os.path.join("realtime_output", "realtime_recorded_output.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# === Data Structures ===
frame_index = 0
recent_predictions = defaultdict(lambda: deque(maxlen=VOTING_WINDOW))
activity_history = defaultdict(
    lambda: {"activity": None, "last_frame": -999, "start_frame": None}
)
all_tracks = {}

print("📷 Starting real-time detection... Press 'q' to quit.")

# === Main Loop ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to open camera.")
        break

    frame_disp = frame.copy()
    results = model(frame)[0]

    detections = []
    for det in results.boxes:
        cls = int(det.cls)
        class_name = model.names[cls]
        conf = float(det.conf)
        if class_name in class_thresholds and conf > class_thresholds[class_name]:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, class_name))

    # === Tracking ===
    tracks = tracker.update_tracks(detections, frame=frame)
    seen_ids = set()

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        class_name = track.get_det_class()
        l, t, w_box, h_box = map(int, track.to_ltrb())
        r, b = l + w_box, t + h_box

        seen_ids.add(track_id)
        all_tracks[track_id] = (l, t, r, b)
        recent_predictions[track_id].append(class_name)
        voted_class = Counter(recent_predictions[track_id]).most_common(1)[0][0]
        color = activity_colors.get(voted_class, (255, 255, 255))
        label = f"{voted_class} | ID {track_id}"

        cv2.rectangle(frame_disp, (l, t), (r, b), color, 2)
        (label_width, label_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            frame_disp,
            (l, t - label_height - 10),
            (l + label_width + 4, t),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            frame_disp,
            label,
            (l + 2, t - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        hist = activity_history[track_id]
        if voted_class != hist["activity"] or frame_index - hist["last_frame"] > 10:
            activity_history[track_id] = {
                "activity": voted_class,
                "last_frame": frame_index,
                "start_frame": frame_index,
            }
        else:
            activity_history[track_id]["last_frame"] = frame_index

    # === Show inactive tracks ===
    for track_id, pos in all_tracks.items():
        if track_id not in seen_ids:
            last_frame = activity_history[track_id]["last_frame"]
            if 10 < (frame_index - last_frame) <= 10 + INACTIVE_DISPLAY_FRAMES:
                l, t, r, b = pos
                color = activity_colors["inactive"]
                label = f"inactive | ID {track_id}"
                cv2.rectangle(frame_disp, (l, t), (r, b), color, 2)
                (label_width, label_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    frame_disp,
                    (l, t - label_height - 10),
                    (l + label_width + 4, t),
                    color,
                    cv2.FILLED,
                )
                cv2.putText(
                    frame_disp,
                    label,
                    (l + 2, t - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

    frame_disp_resized = cv2.resize(frame_disp, (screen_width, screen_height))
    cv2.imshow("📹 Real-time Activity Tracking", frame_disp_resized)
    video_writer.write(frame_disp)
    frame_index += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# === Save Summary JSON and CSV ===
print("\n💾 Saving dashboard data...")

activity_counts = Counter()
activity_durations = defaultdict(list)
time_series_records = []

for track_id, hist in activity_history.items():
    activity = hist["activity"]
    start = hist.get("start_frame", 0)
    end = hist.get("last_frame", start)
    duration = end - start

    activity_counts[activity] += 1
    activity_durations[activity].append(duration)
    time_series_records.append(
        {"track_id": track_id, "activity": activity, "frame": end}
    )

average_durations = {
    act: round(sum(durations) / len(durations), 2)
    for act, durations in activity_durations.items()
}

os.makedirs("dashboard_data", exist_ok=True)
with open("dashboard_data/activity_stats.json", "w") as f_json:
    json.dump(
        {"counts": dict(activity_counts), "durations": average_durations},
        f_json,
        indent=4,
    )

df_time_series = pd.DataFrame(time_series_records)
df_time_series.to_csv("dashboard_data/time_series.csv", index=False)

print(f"\n✅ Video saved to: {output_path}")
print("✅ JSON and CSV saved to: dashboard_data/")
cap.release()
video_writer.release()
cv2.destroyAllWindows()
