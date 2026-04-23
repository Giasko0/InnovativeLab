# Remaining Manual Fill-Ins

## ✅ COMPLETED (no action needed)

### Recycling rules
`hk_recycling_rules.json` — all 15 classes covered with bin, location, and note.
Battery note now uses a generic designated collection-point wording.

### Demo wording
Fixed in `execution_pipeline.py` prompt:
> "Looks like you are showing me a ____. You should throw it in ____."

The Groq Vision model fills in the blanks from the detected label + rule.

### Special-case rules confirmed
| Item | Disposal |
|---|---|
| Greasy / dirty pizza box | General waste |
| Drink carton | Paper bin or GREEN@COMMUNITY carton point |
| Broken glass | General waste (safety first) |
| Battery | Designated battery point |
| Glass bottle | Designated glass collection point |
| Food waste | Designated food waste point |

---

## 🔲 STILL MANUAL — fill in yourself

### 1. Exact local addresses (optional for demo, useful for report)
Only needed if your professor asks for a concrete example.

Find your nearest:
- **General map / nearby facilities**: https://www.wastereduction.gov.hk/en-hk/one-stop-shop
- **GREEN@COMMUNITY locator**: https://www.wastereduction.gov.hk/en-hk/waste-reduction-programme/green-at-community
- **Battery collection points**: https://www.wastereduction.gov.hk/en-hk/waste-reduction-programme/rechargeable-battery-recycling-programme
- **Food waste points/spots**: https://www.wastereduction.gov.hk/en-hk/waste-reduction-programme/food-waste-recycling-schemes

For the demo itself, just say "find your nearest point at the EPD recycling map" — this is already baked into the rules.

### 2. Evaluation results
Fill in after a real run:
- [ ] Latency: Vision → TTS → audio start = ~__ seconds
- [ ] Test items: plastic bottle ✓ / can ✓ / glass bottle ✓ / etc.
- [ ] Any detection confusion: e.g. drink carton misread as corrugated carton

Quick table for report:

| Test item | Expected disposal | Model output | Correct (Y/N) | Latency (s) |
|---|---|---|---|---|
| Plastic bottle | Plastic recycling bin |  |  |  |
| Drink can | Metal recycling bin |  |  |  |
| Glass bottle | Designated glass collection point |  |  |  |
| Drink carton | Dedicated carton route / GREEN@COMMUNITY |  |  |  |
| Cigarette butt | General waste bin |  |  |  |
| Dirty pizza box | General waste bin |  |  |  |
| Plastic film | General waste bin |  |  |  |
| Battery | Designated battery collection point |  |  |  |

### 3. YOLO26n weights path
Change in `demo.py` main() or set env var:
```bash
export TRASHSORT_WEIGHTS=Code/train5/weights/best.pt
```

### 4. Groq API key
```bash
export GROQ_API_KEY=your_key_here
# or put it in Code/.env
```

### 5. Professor-specific scope
Confirm with your professor:
- Is Groq Vision allowed or should it be local-only?
- Do they want a written latency table?
- Do they need the EPD sources cited in a specific format?
