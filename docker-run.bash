docker run -d \
  -e BROKER_URL="mqtt://192.168.200.206:1883" \
  -e IN_TOPIC="cam1/pose/bboxes" \
  -e OUT_TOPIC="bytetrack/tracks" \
  mqtt_bytetrack_cpu_mqtt