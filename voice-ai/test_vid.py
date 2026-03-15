import requests
url = "http://localhost:8000/analyze_video"
files = {'video': open('test.mp3', 'rb')}
r = requests.post(url, files=files)
print(r.status_code)
print(r.json())
