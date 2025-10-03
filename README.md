# Image → Story (HF Space, API-enabled)

**Space**: https://huggingface.co/spaces/nehabathuri/Image-to-Story

## API
POST via Gradio client:
```python
from gradio_client import Client, handle_file
client = Client("nehabathuri/Image-to-Story")
title, story = client.predict(
  handle_file("sample.jpg"),
  "Children", "Adventure", 300, 600, True, 0.9, 0.95,
  api_name="/generate"
)
print(title, story)
