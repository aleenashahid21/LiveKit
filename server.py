"""
Deployed on Render. Combines three jobs into one process:
  1. A background thread that joins the LiveKit Cloud room and runs your
     emotion model on live audio (same logic as main.py / main_polished.py).
  2. A /token endpoint so client.html can fetch a fresh token instead of
     a hardcoded one that expires.
  3. A /status endpoint so display.html can poll the current emotion,
     replacing the local status.json file (which won't reliably persist
     the way you'd want across a cloud host).

Environment variables required (set these in Render's dashboard, not in code):
  LIVEKIT_URL     e.g. wss://your-project.livekit.cloud
  LIVEKIT_API_KEY
  LIVEKIT_API_SECRET
"""

import asyncio
import os
import threading
from collections import Counter, deque

import joblib
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from livekit import rtc, api

from featureextraction import extract_features

LIVEKIT_URL = os.environ["LIVEKIT_URL"]
API_KEY = os.environ["LIVEKIT_API_KEY"]
API_SECRET = os.environ["LIVEKIT_API_SECRET"]
ROOM_NAME = "emotion-demo"

TARGET_SR = 16000
WINDOW_SECONDS = 3
WINDOW_SAMPLES = TARGET_SR * WINDOW_SECONDS
SMOOTHING_WINDOW = 4

model = joblib.load("best_model_tuned_8020.pkl")
emotion_map = {
    0: "female_angry", 1: "female_disgust", 2: "female_fear",
    3: "female_happy", 4: "female_neutral", 5: "female_sad", 6: "female_surprise",
}
EMOJI = {
    "angry": "😠", "disgust": "🤢", "fear": "😨", "happy": "😄",
    "neutral": "😐", "sad": "😢", "surprise": "😲",
}

recent_predictions: deque = deque(maxlen=SMOOTHING_WINDOW)
latest_status = {"label": "neutral", "confidence": 0.0, "emoji": "🎙️"}
status_lock = threading.Lock()

app = Flask(__name__)
CORS(app)  # allows client.html/display.html on Vercel to call this Render URL


def predict_emotion(y: np.ndarray, sr: int = TARGET_SR):
    feats = extract_features(y, sr)
    proba = model.predict_proba([feats])[0]
    idx = int(np.argmax(proba))
    raw_label = emotion_map.get(idx, str(idx))
    return raw_label.replace("female_", ""), float(proba[idx])


def smoothed_label(new_label: str):
    recent_predictions.append(new_label)
    winner, count = Counter(recent_predictions).most_common(1)[0]
    return winner, count / len(recent_predictions)


async def handle_audio_track(track: rtc.Track):
    audio_stream = rtc.AudioStream(track, sample_rate=TARGET_SR, num_channels=1)
    buffer: list[float] = []
    async for event in audio_stream:
        frame = event.frame
        samples = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
        buffer.extend(samples.tolist())

        if len(buffer) >= WINDOW_SAMPLES:
            window = np.array(buffer[:WINDOW_SAMPLES], dtype=np.float32)
            buffer = buffer[WINDOW_SAMPLES:]
            try:
                raw_label, _ = predict_emotion(window)
                label, agreement = smoothed_label(raw_label)
                with status_lock:
                    latest_status.update(
                        label=label, confidence=agreement, emoji=EMOJI.get(label, "🎙️")
                    )
                print(f"[Live] {label} ({agreement:.0%} agreement)")
            except Exception as e:
                print("Live prediction failed:", e)


def build_listener_token() -> str:
    token = (
        api.AccessToken(API_KEY, API_SECRET)
        .with_identity("emotion-listener")
        .with_name("Emotion Listener")
        .with_grants(api.VideoGrants(room_join=True, room=ROOM_NAME))
    )
    return token.to_jwt()


async def run_listener():
    room = rtc.Room()

    @room.on("track_subscribed")
    def handle_track(track: rtc.Track, publication, participant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            print(f"Subscribed to audio from {participant.identity}")
            asyncio.create_task(handle_audio_track(track))

    await room.connect(LIVEKIT_URL, build_listener_token())
    print("Listener connected to LiveKit Cloud room:", ROOM_NAME)
    await asyncio.Future()  # keep alive


def start_listener_thread():
    asyncio.run(run_listener())


# --- HTTP endpoints for the frontend (client.html / display.html) ---------


@app.route("/")
def health():
    return "ok"  # Render's free tier just needs something to respond


@app.route("/token")
def issue_token():
    """client.html calls this on page load to get a fresh join token."""
    identity = request.args.get("identity", "guest")
    token = (
        api.AccessToken(API_KEY, API_SECRET)
        .with_identity(identity)
        .with_grants(api.VideoGrants(room_join=True, room=ROOM_NAME))
    )
    return jsonify({"token": token.to_jwt(), "url": LIVEKIT_URL})


@app.route("/status")
def status():
    """display.html polls this instead of reading a local status.json file."""
    with status_lock:
        return jsonify(dict(latest_status))


if __name__ == "__main__":
    threading.Thread(target=start_listener_thread, daemon=True).start()
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
