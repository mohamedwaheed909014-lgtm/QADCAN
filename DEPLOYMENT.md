# Deployment

This app has a static frontend and a FastAPI backend that also calls the
OpenSCAD command-line tool for STL export. A permanent deployment should host
the backend on a server/container platform. Vercel can host the frontend, but
it is not a good fit for the full backend because OpenSCAD is a native binary
and generation/export calls can be long-running.

## Recommended: Render full-app deployment

1. Push this repository to GitHub.
2. In Render, create a new Web Service from the GitHub repository.
3. Choose Docker environment. Render will use `Dockerfile` automatically.
4. Add at least one model/provider secret:
   - `OPENROUTER_API_KEY`
   - or `OPENAI_API_KEY`
   - or configure an accessible Ollama endpoint.
5. Keep these existing public env vars:
   - `EMBEDDING_BACKEND=tfidf`
   - `OPENSCAD_BIN=openscad`
   - `OPENROUTER_ALLOW_PAID=false`
6. Deploy.

The resulting Render URL, such as:

```text
https://openscad-copilot.onrender.com
```

is the permanent app link to send.

## Vercel-style setup

If you specifically want a `*.vercel.app` frontend link:

1. Deploy the backend first on Render/Railway/Fly with the Dockerfile.
2. Deploy `frontend/` to Vercel as a static site.
3. Add a Vercel rewrite from `/api/:path*` to the backend URL.

Example Vercel rewrite:

```json
{
  "rewrites": [
    {
      "source": "/api/:path*",
      "destination": "https://YOUR-BACKEND-URL.onrender.com/api/:path*"
    }
  ]
}
```

Without a deployed backend, a Vercel frontend will open but generation will show
backend/API loading errors.
