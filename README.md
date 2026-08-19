# LiveKit Speech Emotion Recognition (SER)

A real-time speech emotion recognition demo. It captures live microphone
audio in the browser, streams it through [LiveKit](https://livekit.io/),
runs it through a trained emotion classification model, and displays the
detected emotion (with emoji, color, and confidence) alongside a live
speech-to-text transcript — all on a single page.

**Live demo:** https://live-kit-ashy.vercel.app/display.html

## Features

- **Real-time emotion detection** — classifies speech into 7 emotions
  (angry, disgust, fear, happy, neutral, sad, surprise) using MFCC and
  spectral audio features fed into a trained ML model
- **Live audio-to-text transcription** — captions your speech as you talk
- **Visual feedback** — large emoji, color-coded background, and a
  confidence score ("X% of recent readings agree") that updates live
- **Session controls** — Join Room, Mute, and Leave
- **Session summary** — on Leave, shows a breakdown of emotions detected
  during that session (e.g. "happy: 40%, neutral: 30%, sad: 30%")

## How it works

```
Browser mic  →  LiveKit Cloud (audio streaming)
                     ↓
        Backend listener (Render, Python)
                     ↓
     Feature extraction (MFCCs, spectral centroid/rolloff)
                     ↓
        Trained model → predicted emotion
                     ↓
     /status endpoint  ←  polled every 700ms  →  display.html
```

- The browser (`display.html`) joins a LiveKit room and publishes the
  user's microphone audio.
- A backend process (`server.py`), also connected to the same LiveKit
  room, subscribes to that audio, extracts features, and runs them
  through the trained model every few seconds.
- The backend exposes small HTTP endpoints (`/token`, `/status`,
  `/summary`, `/reset`) that the frontend polls to get a fresh join
  token, the current emotion, and an end-of-session summary.
- Live captions are generated entirely in the browser using the Web
  Speech API — no backend involved for that part.

## Tech stack

| Layer          | Tool / Library                                   |
|-----------------|--------------------------------------------------|
| Audio transport | [LiveKit Cloud](https://livekit.io/)             |
| Frontend        | HTML, JavaScript, LiveKit JS SDK, Web Speech API |
| Backend         | Python, Flask, Flask-CORS                        |
| ML model        | scikit-learn / CatBoost model (`.pkl`), librosa for feature extraction |
| Hosting         | Vercel (frontend, static), Render (backend, Python) |

## Project structure

```
├── display.html            # main app: join/mute/leave, live emotion display, transcript
├── client.html              # earlier standalone client (kept for reference)
├── server.py                 # Flask backend: LiveKit listener + API endpoints
├── featureextraction.py      # audio feature extraction (MFCCs, spectral features)
├── best_model_tuned_8020.pkl # trained emotion classification model
├── compute_accuracy.py       # standalone script to evaluate model accuracy on labeled test data
├── requirements.txt          # backend Python dependencies
├── Procfile                  # process start command for hosting
├── vercel.json                # tells Vercel to serve the HTML files as static
└── .vercelignore              # excludes backend/Python files from the Vercel (frontend) deployment
```

## Running locally

**Backend**
```bash
pip install -r requirements.txt
export LIVEKIT_URL=wss://your-project.livekit.cloud
export LIVEKIT_API_KEY=your_api_key
export LIVEKIT_API_SECRET=your_api_secret
python server.py
```

**Frontend**
Just open `display.html` in a browser, or serve it with any static file
server. Update `BACKEND_URL` at the top of the `<script>` section if
you're pointing it at a different backend.

## Deployment

- **Frontend** is deployed on Vercel as a static site (`vercel.json`
  configures this). `.vercelignore` excludes the Python backend files
  from that deployment.
- **Backend** is deployed on Render as a Python web service, using
  `requirements.txt` and `Procfile`. Environment variables
  (`LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`) are set in
  the Render dashboard, not committed to the repo.

## Notes

- The backend runs on Render's free tier, which sleeps after periods of
  inactivity. A keep-alive ping (e.g. via cron-job.org) is recommended
  so the app responds instantly for anyone opening the link.
- Live captions require a browser that supports the Web Speech API
  (Chrome or Edge recommended).
