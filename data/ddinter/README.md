# DDInter CSV data

Download drug–drug interaction CSV files and place them in this folder.

**Sources (pick one):**

- [Kaggle — DDInter dataset](https://www.kaggle.com/datasets/thedevastator/ddinter-dataset-drug-drug-interactions)
- [Zenodo](https://zenodo.org/records/5549420)
- [DDInter download page](https://ddinter.scbdd.com/download/) (files like `ddinter_downloads_code_*.csv`)

**Expected columns** (flexible naming):

- Drug A / Drug B (or `drug_a`, `drug1`, …)
- Level / severity (optional)
- Mechanism, management (optional)

`interaction_check` loads all `*.csv` files here on first use.

Without CSVs, a small **demo** pair set is used for development (e.g. acetaminophen + ibuprofen).
