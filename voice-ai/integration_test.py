import numpy as np
import soundfile as sf
import cv2
import requests

# 1. Create a 1-second 16kHz sine wave for Audio
sr = 16000
t = np.linspace(0, 1, sr, endpoint=False)
audio = np.sin(2 * np.pi * 440 * t)
sf.write("test_real.wav", audio, sr)

# 2. Test Audio Endpoint
print("Testing Audio Endpoint...")
url = "http://localhost:8000/analyze_audio"
files = {'audio': open('test_real.wav', 'rb')}
r = requests.post(url, files=files)
print("Code:", r.status_code)
print("JSON:", r.json())

# 3. Create a 1-second dummy 30fps video for Video
fps = 30
width, height = 320, 240
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # use mp4v
out = cv2.VideoWriter('test_real.mp4', fourcc, fps, (width, height))
for _ in range(fps):
    frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    out.write(frame)
out.release()

# 4. Test Video Endpoint
print("\nTesting Video Endpoint...")
url = "http://localhost:8000/analyze_video"
files = {'video': open('test_real.mp4', 'rb')}
r = requests.post(url, files=files)
print("Code:", r.status_code)
print("JSON:", r.json())
