# Google Colab setup

Use this only if you want to run the same backend from Colab before deploying to Render.

## Cell 1 — install

```python
!pip -q install -r requirements.txt
!apt-get -qq update && apt-get -qq install -y iverilog
```

## Cell 2 — copy the project into Colab

Upload/extract this project into `/content/verilog-agentic-ai`.

## Cell 3 — configure the API key

In Colab, open the **Secrets** panel, create `GEMINI_API_KEY`, and allow the notebook to access it.

Then run:

```python
from google.colab import userdata
import os
os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")
os.environ["GEMINI_MODEL"] = "gemini-2.5-flash"
```

## Cell 4 — start API

```python
%cd /content/verilog-agentic-ai
!uvicorn app.main:app --host 0.0.0.0 --port 8000
```

For the final project, Render is preferable because the service gets a permanent web URL and the Dockerfile installs Icarus automatically.
