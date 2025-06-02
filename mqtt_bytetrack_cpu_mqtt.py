#!/usr/bin/env python3
import json
import threading
import time
import os
import argparse

import numpy as np
import paho.mqtt.client as mqtt
from boxmot import ByteTrack

def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time tracking publisher: subscribes to pose bounding boxes, "
                    "runs ByteTrack, and republishes track results over MQTT."
    )
    parser.add_argument(
        "--mqtt_host",
        default=os.getenv("MQTT_HOST", "192.168.200.206"),
        help="MQTT broker hostname or IP"
    )
    parser.add_argument(
        "--mqtt_port",
        type=int,
        default=int(os.getenv("MQTT_PORT", "1883")),
        help="MQTT broker port"
    )
    parser.add_argument(
        "--in_topic",
        default=os.getenv("IN_TOPIC", "cam1/pose/bboxes"),
        help="Incoming MQTT topic for raw bounding boxes"
    )
    parser.add_argument(
        "--out_topic",
        default=os.getenv("OUT_TOPIC", "bytetrack/tracks"),
        help="Outgoing MQTT topic for track results"
    )
    return parser.parse_args()

# where dets_array is shape (N,6): [x1, y1, x2, y2, score, class_id]
dets_buffer: dict[str, tuple[np.ndarray, float]] = {}

# ByteTrack tracker instance (retains internal state across frames)
tracker = ByteTrack()

# Single shared MQTT client (initialized in main)
mqtt_client: mqtt.Client

# Lock to protect dets_buffer
buffer_lock = threading.Lock()

# ────────────────────────────────────────────────────────────────────────────────
# MQTT CALLBACKS
# ────────────────────────────────────────────────────────────────────────────────
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[MQTT] Connected to broker, subscribing to '{userdata['in_topic']}'")
        client.subscribe(userdata["in_topic"], qos=0)
    else:
        print(f"[MQTT] Connection failed (code={rc})")

def on_message(client, userdata, msg):
    """
    Called whenever a new message arrives on the subscribed inbound topic.
    Expects JSON payload:
      {
        "frame_id": "…FRAME:000110",
        "single_gpu_fps": 6.23,
        "scale": 1.0,
        "bboxes": [
          [x1_s, y1_s, x2_s, y2_s], …
        ],
        "bboxe_scores": [s1, s2, …]
      }
    """
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        fid_str = payload.get("frame_id", "")
        # Keep frame_id as string (e.g. "FRAME:000110")
        frame_id = fid_str

        scale = float(payload.get("scale", 1.0))
        raw_boxes = payload.get("bboxes", [])         # List of [x1_s, y1_s, x2_s, y2_s]
        raw_scores = payload.get("bboxe_scores", [])  # List of floats

        # Build an (N×6) float32 array: [x1, y1, x2, y2, score, class_id]
        det_list = []
        for (box, score) in zip(raw_boxes, raw_scores):
            x1_s, y1_s, x2_s, y2_s = box
            # Undo the “scale” to get pixel coords
            x1 = x1_s / scale
            y1 = y1_s / scale
            x2 = x2_s / scale
            y2 = y2_s / scale
            det_list.append([x1, y1, x2, y2, score, 0])  # class_id=0

        if det_list:
            dets_arr = np.asarray(det_list, dtype=np.float32)
        else:
            dets_arr = np.zeros((0, 6), dtype=np.float32)

        with buffer_lock:
            dets_buffer[frame_id] = (dets_arr, time.time())
            # As soon as we have new detections for frame_id, run tracking:
            process_tracking(frame_id, userdata["out_topic"])

    except Exception as e:
        print(f"[MQTT] Failed to parse incoming bboxes message: {e}")

# ────────────────────────────────────────────────────────────────────────────────
# TRACKING + PUBLISHING
# ────────────────────────────────────────────────────────────────────────────────
def process_tracking(frame_id: str, out_topic: str):
    """
    If dets_buffer contains this frame_id, pop it, run ByteTrack.update(),
    then publish only (x1,y1,x2,y2,track_id,conf) to OUT_TOPIC.
    """
    # Pop detections (no frame buffer used, so process once)
    dets, _ = dets_buffer.pop(frame_id, (None, None))
    if dets is None:
        return

    # ByteTrack.update expects (dets, frame). Use a dummy 1×1 black frame.
    dummy_frame = np.zeros((1, 1, 3), dtype=np.uint8)

    # Run the tracker: returns list/array of rows:
    # [x1, y1, x2, y2, track_id, conf, cls_id, det_index]
    tracklets = tracker.update(dets, dummy_frame)

    track_ids = []
    bboxes = []
    track_scores = []

    for t in tracklets:
        x1, y1, x2, y2, tid, conf, *_ = t
        track_ids.append(int(tid))
        bboxes.append([int(x1), int(y1), int(x2), int(y2)])
        track_scores.append(float(conf))

    message = {
        "frame_id": frame_id,
        "track_ids": track_ids,
        "bboxes": bboxes,
        "track_scores": track_scores,
    }

    payload = json.dumps(message)
    mqtt_client.publish(out_topic, payload)

# ────────────────────────────────────────────────────────────────────────────────
# MAIN ENTRYPOINT
# ────────────────────────────────────────────────────────────────────────────────
def main():
    global mqtt_client

    args = parse_args()

    # Store topics in userdata so callbacks can access them
    userdata = {
        "in_topic": args.in_topic,
        "out_topic": args.out_topic
    }

    mqtt_client = mqtt.Client(userdata=userdata)
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message

    mqtt_client.connect(args.mqtt_host, args.mqtt_port, keepalive=60)
    mqtt_client.loop_start()
    print(f"[MAIN] MQTT loop started. Listening on '{args.in_topic}'…")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("[MAIN] Interrupted by user, shutting down…")
    finally:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        print("[MAIN] Exited gracefully.")

if __name__ == "__main__":
    main()
