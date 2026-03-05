# Best Guess (Streamlit App)

A browser app that recommends the best guess in a 1-100 number-line game with 3 players.
It uses optimal-response logic and a tie-break rule that prefers the midpoint of the largest gap.

## Local Run

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then open `http://localhost:8501`.

## Deploy to Streamlit Community Cloud

No command line is required for this flow.

1. Create a new GitHub repository from the browser.
2. Use **Add file** > **Upload files** and upload these files to the repository root:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - Optional: `.streamlit/config.toml`
3. Go to [https://share.streamlit.io](https://share.streamlit.io).
4. Click **New app** and select your repository.
5. Set **Main file path** to `app.py`.
6. Click **Deploy**.

After deployment, you get a public URL you can open from any device.
