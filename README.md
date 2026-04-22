# Computer Graphic Project

Interactive 3D solar system project with two runtimes:

- `main.py` is the original Python + Pygame + OpenGL desktop version.
- `index.html` / `app.js` is the browser version prepared for Vercel deployment.

## Deploy to Vercel

Import this repository into Vercel and keep the default settings. The project is static, so no build command is required. Vercel will serve `index.html` and the local texture assets from `assets/`.

## Run Locally

Preview the Vercel version with a small static server:

```bat
python -m http.server 4173
```

Then open `http://127.0.0.1:4173`.

For the desktop OpenGL version:

```bat
RUN.bat
```
