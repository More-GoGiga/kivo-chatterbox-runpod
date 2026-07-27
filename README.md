# KIVO Chatterbox Worker

Queue-native Runpod Serverless worker for Chatterbox Multilingual V3.

## Request

```json
{
  "input": {
    "text": "Hello from KIVO.",
    "language": "en",
    "exaggeration": 0.5,
    "cfg_weight": 0.5
  }
}
```

`reference_audio` is optional and accepts either raw base64 audio or an audio
data URI for voice cloning. Without it, Chatterbox uses its default voice.

The response contains a base64-encoded PCM WAV in `output.audio_base64`.
