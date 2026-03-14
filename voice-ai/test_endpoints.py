import os
import time
import requests
import wave
import struct

# Create a dummy WAV file
def create_dummy_wav(filename="dummy_test.wav"):
    sample_rate = 16000
    duration_seconds = 2
    num_samples = sample_rate * duration_seconds
    
    with wave.open(filename, 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        
        # Write silence
        for _ in range(num_samples):
            f.writeframes(struct.pack('<h', 0))

def test_analyze_audio(filename):
    url = "http://127.0.0.1:8000/analyze_audio"
    start_time = time.time()
    with open(filename, 'rb') as f:
        files = {'audio': (filename, f, 'audio/wav')}
        response = requests.post(url, files=files)
    latency = time.time() - start_time
    print(f"[{latency:.2f}s] POST /analyze_audio -> {response.status_code}")
    print(response.json())
    return latency

def test_enroll_speaker(filename, user_id="test_user"):
    url = "http://127.0.0.1:8000/enroll_speaker"
    start_time = time.time()
    with open(filename, 'rb') as f:
        files = {'audio': (filename, f, 'audio/wav')}
        data = {'user_id': user_id}
        response = requests.post(url, files=files, data=data)
    latency = time.time() - start_time
    print(f"[{latency:.2f}s] POST /enroll_speaker -> {response.status_code}")
    print(response.json())
    return latency

def test_verify_speaker(filename, user_id="test_user"):
    url = "http://127.0.0.1:8000/verify_speaker"
    start_time = time.time()
    with open(filename, 'rb') as f:
        files = {'audio': (filename, f, 'audio/wav')}
        data = {'user_id': user_id}
        response = requests.post(url, files=files, data=data)
    latency = time.time() - start_time
    print(f"[{latency:.2f}s] POST /verify_speaker -> {response.status_code}")
    print(response.json())
    return latency

if __name__ == "__main__":
    wav_file = "dummy_test.wav"
    create_dummy_wav(wav_file)
    
    try:
        t1 = test_analyze_audio(wav_file)
        t2 = test_enroll_speaker(wav_file)
        t3 = test_verify_speaker(wav_file)
        
        if all(t < 2.0 for t in [t1, t2, t3]):
            print("\nPerformance Validation: SUCCESS (All endpoints < 2s)")
        else:
            print("\nPerformance Validation: WARNING (Some endpoints took >= 2s)")
    except Exception as e:
        print(f"Endpoint test failed: {e}")
    finally:
        if os.path.exists(wav_file):
            os.remove(wav_file)
