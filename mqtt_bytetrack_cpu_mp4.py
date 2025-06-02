#!/usr/bin/env python3
"""
Real-time MQTT-to-ByteTrack bridge **with ordered frame buffering**
——————————————————————————————————————————————————————————————
Updates in this revision:
• Save first 100 frames (1920×1080) with bounding‐boxes overlaid to output.mp4.
  – Creates a cv2.VideoWriter the first time a frame is processed.
  – Draws each tracklet’s bounding‐box onto a blank 1920×1080 image and writes it.
  – Once 100 frames have been written, releases the writer and stops writing.
Everything else is identical to the previously shared version.
"""
import json, re, threading, time, os, argparse
from typing import Dict, Tuple, Optional

import numpy as np
import cv2
import paho.mqtt.client as mqtt
from boxmot import ByteTrack

# ────────────────────────────  PARAMETERS  ──────────────────────────── #
SKIP_AHEAD    = 3      # how many later frames must arrive before we skip a hole
FRAME_TIMEOUT = 2.0    # seconds before we drop a never-arrived frame
FRAME_ID_RE   = re.compile(r"(\D*)(\d+)$")  # prefix + numeric suffix extractor

# Video‐writing parameters:
VIDEO_FILENAME   = "output.mp4"
IMAGE_WIDTH      = 1920
IMAGE_HEIGHT     = 1080
FPS              = 30              # frames per second for the output video
MAX_FRAMES       = 100             # only save first 100 frames
# ---------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser("ByteTrack MQTT bridge with ordered buffering")
    p.add_argument("--broker",    default=os.getenv("BROKER_URL", "mqtt://127.0.0.1:1883"))
    p.add_argument("--in_topic",  default=os.getenv("IN_TOPIC",   "cam1/pose/bboxes"))
    p.add_argument("--out_topic", default=os.getenv("OUT_TOPIC",  "bytetrack/tracks"))
    return p.parse_args()

# Buffer keyed by *integer* frame number  →  (dets_array, t_arrived, original_id)
# Guarded by `buffer_lock` for thread-safety inside the MQTT callback.
dets_buffer: Dict[int, Tuple[np.ndarray, float, str]] = {}
next_seq: Optional[int] = None          # frame number we expect to process next
buffer_lock = threading.Lock()

# ByteTrack instance (re-created whenever sequence restarts)
tracker = ByteTrack()

# Video‐writer globals:
video_writer: Optional[cv2.VideoWriter] = None
saved_frame_count = 0

mqtt_client: mqtt.Client  # set later in main()

# ─────────────────────────  HELPERS FOR IDS  ────────────────────────── #
def split_frame_id(fid: str) -> tuple[str,int]:
    """
    ("FRAME", 123)  from  "FRAME:000123"
    """

    out = fid.split(':')
    if len(out) != 2:
        return "", -1 # invalid format, return empty prefix and zero ID
    prefix, frame_id_str = fid.split(':')
    return prefix, int(frame_id_str)

# ─────────────────────────  MQTT CALLBACKS  ─────────────────────────── #
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[MQTT] Connected → subscribing '{userdata['in_topic']}'")
        client.subscribe(userdata["in_topic"], qos=2)
    else:
        print(f"[MQTT] Connection failed ({rc})")

def on_message(client, userdata, msg):
    """MQTT message handler: parse, buffer, and trigger `pump_buffer()`."""
    global next_seq, tracker

    payload = json.loads(msg.payload.decode())
    fid = payload.get("frame_id", "")
    prefix, seq = split_frame_id(fid)

    # ——— Handle RESET when frame 0 arrives ———————————————————— #
    if seq == 0:
        print("[RESET] Received frame_id 0 – resetting tracker and buffers")
        tracker = ByteTrack()          # fresh state, track IDs restart
        with buffer_lock:
            dets_buffer.clear()
            next_seq = 0               # start sequence anew at 0
    # ———————————————————————————————————————————————— #

    scale  = float(payload.get("scale", 1.0))
    boxes  = payload.get("bboxes", [])
    scores = payload.get("bboxe_scores", [])
    dets = np.array(
        [[x1/scale, y1/scale, x2/scale, y2/scale, s, 0]
            for (x1, y1, x2, y2), s in zip(boxes, scores)],
        dtype=np.float32
    ) if boxes else np.zeros((0, 6), np.float32)

    with buffer_lock:
        dets_buffer[seq] = (dets, time.time(), fid)
        if next_seq is None:
            next_seq = seq  # initialise cursor on the very first packet

    pump_buffer(prefix, len(str(seq)), userdata["out_topic"])


# ──────────────────────────  BUFFER PUMP  ───────────────────────────── #
def pump_buffer(prefix: str, width: int, out_topic: str):
    """Process frames strictly in order, optionally skipping holes."""
    global next_seq

    while True:
        with buffer_lock:
            if next_seq is None:
                return  # nothing initialised yet

            entry = dets_buffer.get(next_seq)
            if entry:
                dets, _, fid = dets_buffer.pop(next_seq)
            else:
                # Decide whether to skip a missing frame
                newer = [s for s in dets_buffer if s > next_seq]
                if len(newer) >= SKIP_AHEAD or (
                    newer and time.time() - min(dets_buffer[s][1] for s in newer) > FRAME_TIMEOUT):
                    print(f"[BUF] Skipping missing frame {next_seq}")
                    next_seq += 1
                    continue  # try again with incremented cursor
                break  # cannot skip yet – leave for later

        if entry:
            process_tracking(fid, dets, out_topic)
            next_seq += 1
        else:
            break  # nothing ready right now

        # House-keeping: drop stale buffered items
        with buffer_lock:
            stale = [s for s, (_, t, _) in dets_buffer.items()
                     if time.time() - t > FRAME_TIMEOUT]
            for s in stale:
                dets_buffer.pop(s, None)

# ───────────────────────  TRACKING & PUBLISHING  ────────────────────── #
def process_tracking(frame_id: str, dets: np.ndarray, out_topic: str):
    global video_writer, saved_frame_count

    # Create a dummy frame for ByteTrack (we don’t actually have a real image here)
    dummy_frame = np.zeros((1, 1, 3), np.uint8)
    tracklets = tracker.update(dets, dummy_frame)

    # Publish over MQTT
    mqtt_client.publish(out_topic, json.dumps({
        "frame_id": frame_id,
        "track_ids":   [int(t[4]) for t in tracklets],
        "bboxes":      [[int(t[0]), int(t[1]), int(t[2]), int(t[3])] for t in tracklets],
        "track_scores":[float(t[5]) for t in tracklets],
    }))

    # --- Write first 100 frames to an MP4 ---
    if saved_frame_count < MAX_FRAMES:
        # Initialize VideoWriter on the first time we’re here
        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                VIDEO_FILENAME, fourcc, FPS,
                (IMAGE_WIDTH, IMAGE_HEIGHT)
            )
            if not video_writer.isOpened():
                print("[VIDEO] ERROR: Could not open VideoWriter")
                return

        # Create a blank 1920×1080 image (black background)
        frame = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)

        # Draw each tracklet’s bounding box onto the blank frame
        # tracklet format: [x1, y1, x2, y2, track_id, score]
        for t in tracklets:
            x1, y1, x2, y2, tid, score, _, _ = t
            # Convert to ints and draw rectangle + ID text
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2), int(y2))
            track_id = int(t[4])
            # Draw bounding box (green, thickness=2)
            cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
            # Put track ID above the box
            cv2.putText(
                frame,
                f"ID:{track_id}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                lineType=cv2.LINE_AA
            )

        # Write this frame into the MP4
        video_writer.write(frame)
        saved_frame_count += 1

        # If we just wrote the 100th frame, release the writer
        if saved_frame_count == MAX_FRAMES:
            video_writer.release()
            print(f"[VIDEO] Saved {MAX_FRAMES} frames into '{VIDEO_FILENAME}'")

# ───────────────────────────────  MAIN  ─────────────────────────────── #
def main():
    global mqtt_client

    args = parse_args()
    proto, rest = args.broker.split("://", 1)
    host, port = rest.split(":")
    userdata = {"in_topic": args.in_topic, "out_topic": args.out_topic}

    mqtt_client = mqtt.Client(userdata=userdata)
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message

    mqtt_client.connect(host, int(port), keepalive=60)
    mqtt_client.loop_start()
    print(f"[MAIN] Listening on '{args.in_topic}' → publishing '{args.out_topic}'")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[MAIN] Ctrl-C - exiting")
    finally:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        # If we exited before writing 100 frames, still release writer
        global video_writer
        if video_writer is not None and video_writer.isOpened():
            video_writer.release()
            print(f"[VIDEO] Released VideoWriter (wrote {saved_frame_count} frames total)")

if __name__ == "__main__":
    main()
