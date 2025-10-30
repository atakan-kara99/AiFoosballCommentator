FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        xvfb \
        x11-xserver-utils \
        libgl1-mesa-glx \
        xauth && \
    rm -rf /var/lib/apt/lists/*

ENV DISPLAY=:99
ENV PYTHONUNBUFFERED=1
ENV XAUTHORITY=/root/.Xauthority

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

CMD rm -f /tmp/.X99-lock && \
    Xvfb :99 -screen 0 1280x1024x24 & \
    sleep 2 && \
    touch /root/.Xauthority && \
    xauth generate :99 . trusted && \
    python pipeline.py
