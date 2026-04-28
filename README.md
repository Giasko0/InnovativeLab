# TrashSort

## 1. Install

```bash
python -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```

## 2. Train model

```bash
python training_pipeline.py
```

Output model (default):

`<output-dir>/runs/train_py/weights/best.pt`

## 3. Run inference demo

Set API key (or create `.env` in this folder):

```bash
export GROQ_API_KEY=your_key_here
```

Run:

```bash
python execution_pipeline.py
```

You can resize the inference window while it is running. To start larger:

```bash
python execution_pipeline.py --window-width 1280 --window-height 720
```

Optional custom weights:

```bash
python execution_pipeline.py --weights /path/to/best.pt
```
