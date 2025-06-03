#!/usr/bin/env python3
"""
Real-time MQTT-to-ByteTrack bridge **with ordered frame buffering**
——————————————————————————————————————————————————————————————
Updates in this revision
───────────────────────
• **Track-ID reset on frame 0**
  When a payload arrives whose numeric frame_id is 0 we:
      - clear the frame buffer
      - reset `next_seq`
      - create a fresh `ByteTrack()` instance so new tracks start from ID 1 again
  (This mirrors common camera pipelines that restart numbering with FRAME:000000.)

Everything else is identical to the previously shared version.
"""
import json, re, threading, time, os, argparse
from typing import Dict, Tuple, Optional

import numpy as np
import paho.mqtt.client as mqtt
from boxmot import ByteTrack
from boxmot.trackers.bytetrack.bytetrack import STrack

# ────────────────────────────  PARAMETERS  ──────────────────────────── #
SKIP_AHEAD    = 7      # how many later frames must arrive before we skip a hole
FRAME_TIMEOUT = 2.0    # seconds before we drop a never-arrived frame
FRAME_ID_RE   = re.compile(r"(\D*)(\d+)$")  # prefix + numeric suffix extractor
# ---------------------------------------------------------------------- #


def parse_args():
    p = argparse.ArgumentParser("ByteTrack MQTT bridge with ordered buffering")
    p.add_argument("--broker",    default=os.getenv("BROKER_URL", "mqtt://192.168.200.206:1883"))
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

mqtt_client: mqtt.Client  # set later in main()

# ─────────────────────────  HELPERS FOR IDS  ────────────────────────── #
def split_frame_id(fid: str) -> tuple[str,int]:
    """
    ("FRAME", 123)  from  "FRAME:000123"
    """

    splits = fid.split(':')
    if len(splits) != 2:
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

    try:
        payload = json.loads(msg.payload.decode())
        fid = payload.get("frame_id", "")
        prefix, seq = split_frame_id(fid)

        # ——— Reset logic ———
        reset_now = (seq == 1 or seq == 0)
        if reset_now:
            print("[RESET] frame_id 0 received – clearing buffers & restarting tracker")

            tracker = ByteTrack()
            STrack.clear_count()  # reset track ID counter
            with buffer_lock:
                dets_buffer.clear()
                next_seq = None

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

    except Exception as e:
        print(f"[MQTT] Bad message: {e}")


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
                break  # cannot skip yet - leave for later

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
    dummy_frame = np.zeros((1, 1, 3), np.uint8)  # ByteTrack requires an image
    tracklets = tracker.update(dets, dummy_frame)

    mqtt_client.publish(out_topic, json.dumps({
        "frame_id": frame_id,
        "track_ids":   [int(t[4]) for t in tracklets],
        "bboxes":      [[int(t[0]), int(t[1]), int(t[2]), int(t[3])] for t in tracklets],
        "track_scores":[float(t[5]) for t in tracklets],
    }))


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
        mqtt_client.loop_stop(); mqtt_client.disconnect()


if __name__ == "__main__":
    main()
