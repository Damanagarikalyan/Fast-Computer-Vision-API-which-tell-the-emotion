from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import tempfile
from deepface import DeepFace
from collections import Counter
import asyncio
import nest_asyncio
import uvicorn

# Apply the nest_asyncio patch
nest_asyncio.apply()

app = FastAPI()

EMOTIONS = ['happy', 'sad', 'neutral', 'angry', 'surprise', 'fear', 'disgust']

@app.get("/")
def read_root():
    return {"message": "Welcome to the Emotion Analysis API"}

def extract_frames(video_path, frame_rate=1):
    video_capture = cv2.VideoCapture(video_path)
    frames = []
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / frame_rate)

    success, frame = video_capture.read()
    count = 0
    while success:
        if count % frame_interval == 0:
            frames.append(frame)
        success, frame = video_capture.read()
        count += 1

    video_capture.release()
    return frames

async def analyze_frame_async(frame):
    try:
        loop = asyncio.get_event_loop()
        # Call DeepFace.analyze directly
        analysis = await loop.run_in_executor(None, DeepFace.analyze, frame, ['emotion'])
        return analysis[0]['dominant_emotion']
    except Exception as e:
        print(f"Error processing frame: {e}")
        return None

async def analyze_frames(frames):
    emotion_counter = Counter()
    total_frames = len(frames)

    tasks = [analyze_frame_async(frame) for frame in frames]
    results = await asyncio.gather(*tasks)

    for result in results:
        if result:
            emotion_counter[result] += 1
        else:
            total_frames -= 1

    return emotion_counter, total_frames

def calculate_emotion_percentages(emotion_counter, total_frames):
    percentages = {emotion: (count / total_frames) * 100 for emotion, count in emotion_counter.items()}
    return percentages

@app.post("/analyze_emotion")
async def analyze_emotion(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.file.read())
        video_path = tmp.name

    frames = extract_frames(video_path)
    emotion_counter, total_frames = await analyze_frames(frames)

    if total_frames > 0:
        percentages = calculate_emotion_percentages(emotion_counter, total_frames)
        return JSONResponse(content=percentages)
    else:
        return JSONResponse(content={"error": "No frames processed successfully"}, status_code=500)

# Run the FastAPI server locally
if __name__ == "__main__":
    try:
        port = 8000
        print(f"Running server on http://localhost:{port}")
        uvicorn.run(app, host="0.0.0.0", port=port)
    except Exception as e:
        print(f"Error: {e}")
