# Deploy to Render

This project runs as a Python/FastAPI web service on Render. The frontend is
served by the backend, so deploy it as a Web Service, not a Static Site.

## 1. Push the project to GitHub

Render deploys from a Git repository. Create a GitHub repository and push this
project to it.

Do not commit `.env`; add secrets in the Render dashboard instead.

## 2. Create the Render service

1. Open the Render dashboard.
2. Select **New**.
3. Select **Blueprint** if Render detects `render.yaml`, or select **Web Service**.
4. Connect the GitHub repository.
5. Use these commands if entering settings manually:

```bash
pip install -r requirements.txt
```

```bash
cd backend && uvicorn app:app --host 0.0.0.0 --port $PORT
```

## 3. Add environment variables

Add at least one hosted LLM provider key:

```text
OPENROUTER_API_KEY=your_key_here
```

or:

```text
OPENAI_API_KEY=your_key_here
```

Recommended Render values:

```text
EMBEDDING_BACKEND=tfidf
OPENROUTER_ALLOW_PAID=false
ENABLE_CLARIFICATION=false
```

Avoid `OLLAMA_BASE_URL` on Render unless you also deploy an Ollama server.

## 4. Deploy

After the first deploy finishes, Render will give you a public URL like:

```text
https://openscad-copilot.onrender.com
```

Free Render web services can spin down after idle time, so the first request
after a quiet period may take about a minute.

## Notes

- `backend/users.db` and `backend/logs/*` are local files. On Render free
  instances, filesystem changes should be treated as temporary.
- `OPENSCAD_BIN=openscad` will only work on Render if OpenSCAD is installed in
  the service image. Text generation can still work without it, but STL export
  and server-side OpenSCAD validation may be limited.
