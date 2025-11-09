# STOCKZFNL

Small project with Keras models and example notebooks.

Quick steps to publish this project to GitHub:

1. (Optional) Decide how to handle large model files (.h5/.keras):
   - Use Git LFS to track these files: `git lfs install` and `git lfs track "*.h5" "*.keras"`.
   - Or exclude them via `.gitignore` and upload models elsewhere (Releases, cloud storage).
2. Initialize the repo locally:
   - `git init`
   - `git add .`
   - `git commit -m "Initial commit"`
3. Create a GitHub repo (via website or `gh`) and add it as remote:
   - `git remote add origin https://github.com/<youruser>/<repo>.git`
4. Push to GitHub:
   - `git push -u origin main`

If you want, I can run the local git steps for you from this machine (I won't push without your GitHub credentials). If you prefer, follow the steps above in PowerShell.

Notes
- Model files in this repo may be large. GitHub's 100 MB per-file limit applies when pushing — use Git LFS or upload model files to Releases.
- After pushing, add a short project description, usage instructions, and any license.
