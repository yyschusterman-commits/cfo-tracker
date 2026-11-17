# CFO Buy Tracker
Forward test of the "follow the CFO" insider-buy rule. `tracker.py` runs every weekday evening in GitHub Actions
(`.github/workflows/nightly.yml`), updates `data.json`, and the page (`index.html`) reads it. Run it by hand from the Actions tab ("Run workflow").
