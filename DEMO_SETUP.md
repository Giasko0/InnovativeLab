# TrashSort Demo Setup

## 1. Prepare the model

- Train or copy your YOLO26n weights to a local file.

## 2. Run the demo

```bash
python Code/demo.py
```

## 3. What to demo

- clear plastic bottle -> plastic recycling bin
- drink can -> metal recycling bin
- glass bottle -> designated glass point
- food waste -> designated food waste point
- cigarette butt -> general waste
- dirty pizza box -> general waste

## 4. Useful notes

- Wait for the item to stay stable in frame.
- The demo is biased toward quick, safe advice.
- General waste is the fallback when uncertain or dirty.
