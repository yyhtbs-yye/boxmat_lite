# Base image: Nvidia PyTorch https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM python:3.9-slim

ENV BROKER_URL="mqtt://192.168.200.206:1883" \
    IN_TOPIC="cam1/pose/bboxes" \
    IN_TOPIC="bytetrack/tracks" 
    
# Update and install necessary packages
RUN apt update && apt install -y git

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libgl1-mesa-glx \ 
      libglib2.0-0


# Set the parent working directory
WORKDIR /app

# Clone the repository with submodules into a subdirectory 'boxmot'
RUN git clone https://github.com/yyhtbs-yye/boxmat_lite

RUN cd boxmat_lite && pip install -r requirements.txt \
    && pip install -e .

COPY mqtt_bytetrack_cpu_mqtt.py .

CMD ["bash", "-lc", "\
    python mqtt_bytetrack_cpu_mqtt.py \
      --broker \"${BROKER_URL}\" \
      --in_topic \"${IN_TOPIC}\" \
      --out_topic \"${OUT_TOPIC}\" \
"]


# ------------------------------------------------------------------------------

# A Docker container exits when its main process finishes, which in this case is bash.
# To avoid this, use detach mode.

# Run interactively with all GPUs accessible:
#   docker run -it --gpus all mikel-brostrom/boxmot bash

# Run interactively with specific GPUs accessible (e.g., first and third GPU):
#   docker run -it --gpus '"device=0,2"' mikel-brostrom/boxmot bash

# Run in detached mode (if you exit the container, it won't stop):
# Create a detached Docker container from an image:
#   docker run -it --gpus all -d mikel-brostrom/boxmot

# Access the running container:
#   docker exec -it <container_id> bash

# When you are done with the container, stop it by:
#   docker stop <container_id>
