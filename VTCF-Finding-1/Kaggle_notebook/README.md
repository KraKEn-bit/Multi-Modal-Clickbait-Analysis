# Kaggle Notebook — text_only re-run

## What's in this folder

| Path | Purpose |
|---|---|
| `vctf-research_kag.ipynb` | Clean notebook — **text_only re-train + evaluate only** |
| `vtcf-code/` | Fixed Python code (upload as a Kaggle dataset) |

## Before running on Kaggle

### 1. Upload `vtcf-code/` as a new dataset

1. Zip the folder (keeps structure):

```powershell
Set-Location "E:\Research Paper\Research-2\vtcf-research\Kaggle_notebook"
tar -a -cf vtcf-code.zip vtcf-code
```

2. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) → **New Dataset**
3. Upload `vtcf-code.zip`
4. Name it **`vtcf-code`**
5. Create

### 2. (Optional) Upload vision + full checkpoints

If starting a **fresh** Kaggle session (no existing checkpoints in `/kaggle/working`), upload a dataset containing:

- `best_model_vision_only.pt`
- `best_model_full.pt`

Name it **`vtcf-checkpoints`**. The notebook copies them automatically.

### 3. Update the notebook on Kaggle

1. Open [your notebook](https://www.kaggle.com/code/anybodyk/vctf-research)
2. **File → Import Notebook** → upload `vctf-research_kag.ipynb` (or paste cells)
3. **Add Input** in the sidebar:
   - `vtcf-code` (new)
   - `vtcf-bangla-clickbait-frames` (existing)
   - `vtcf-checkpoints` (optional, if fresh session)
4. Settings: **GPU T4**, **Internet On**
5. **Run All**

## What was removed from the old notebook

- Duplicate diagnostic cells (cells 2–5)
- `vision_only` and `full` training (already done)
- Inline `evaluate.py` patch (fix is in `vtcf-code` now)
- Zip/split/download cells (one-time workaround — checkpoints already downloaded locally)

## Expected training output

```
contrast=0.0000   ← must be zero
val_f1 → ~0.98    ← by epoch 3–5
[text_only debug] batch=0 det_loss=... contrast_loss=0.0000
```
