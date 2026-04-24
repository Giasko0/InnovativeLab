# TrashSort (2-file pipeline + benchmark)

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Train model

```bash
python training_pipeline.py
```

Output model (default):

`datasets/taco_hk_yolo26/runs/train_py/weights/best.pt`

## 3. Run inference demo

Set API key (or create `.env` in this folder):

```bash
export GROQ_API_KEY=your_key_here
```

Run:

```bash
python execution_pipeline.py
```

Optional custom weights:

```bash
python execution_pipeline.py --weights /path/to/best.pt
```

## 4. Compare model quality

Compares:
- reference notebook model: `Alessandro/train5/weights/best.pt`
- your model: `datasets/taco_hk_yolo26/runs/train_py/weights/best.pt`
- same validation set: `datasets/taco_hk_yolo26/data.yaml`

Run:

```bash
python compare_models.py
```

Output report:

`model_comparison.json`
