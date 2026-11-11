"""Score a VietOCR model on the GT-box crops of one split (plan.md E-C1, options C2 / C3).

Data: dataset/pres/vietocr (made by tools/prepare_vietocr_pres.py).
Metrics: CER, WER, exact-match accuracy, overall and per class (tools/ocr_metrics.py),
plus seconds per crop.

Usage:
    # C2: official pretrained vgg_transformer, no fine-tuning
    python vietocr/eval_vietocr.py --weights pretrained --name C2_pretrained
    # C3: fine-tuned weights downloaded from Kaggle (use the config.yml saved with them)
    python vietocr/eval_vietocr.py --weights weights/vietocr_vaipe.pth \
        --config weights/vietocr_config.yml --name C3_finetune
Output (in --out, default results/E-C1): <name>_metrics.json, <name>_predictions.tsv
"""
import argparse
import json
import sys
import time
import types
from pathlib import Path

# vietocr imports imgaug at module level, but only the albumentations augmenter is used.
# imgaug 0.4.0 fails to import on NumPy 2, so stub it.
try:
    import imgaug  # noqa: F401
except Exception:
    _stub = types.ModuleType('imgaug')
    _stub.augmenters = types.ModuleType('imgaug.augmenters')
    sys.modules['imgaug'] = _stub
    sys.modules['imgaug.augmenters'] = _stub.augmenters

import numpy as np  # noqa: E402

# vietocr 0.3.13 reads its lmdb data with np.fromstring(bytes), which recent NumPy rejects.
_np_fromstring = np.fromstring


def _fromstring(string, dtype=float, count=-1, *, sep='', **kw):
    if sep == '' and isinstance(string, (bytes, bytearray, memoryview)):
        return np.frombuffer(string, dtype=dtype, count=count).copy()
    return _np_fromstring(string, dtype=dtype, count=count, sep=sep, **kw)


np.fromstring = _fromstring

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from vietocr.tool.config import Cfg  # noqa: E402
from vietocr.tool.predictor import Predictor  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / 'tools'))
from ocr_metrics import evaluate, format_table  # noqa: E402

PRETRAINED_URL = 'https://vocr.vn/data/vietocr/vgg_transformer.pth'


def read_split(data_root, split):
    """[(rel_path, text, label or None)] for one split."""
    data_root = Path(data_root)
    meta = data_root / f'{split}_meta.tsv'
    if meta.exists():
        rows = []
        with open(meta, encoding='utf-8') as f:
            next(f)
            for line in f:
                line = line.rstrip('\n')
                if line:
                    rel, label, text = line.split('\t', 2)
                    rows.append((rel, text, label))
        return rows
    rows = []
    with open(data_root / f'{split}.txt', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line:
                rel, text = line.split('\t', 1)
                rows.append((rel, text, None))
    return rows


def predict_all(predictor, images, batch=64):
    """Batched prediction; images sorted by width so each batch has one or few sizes."""
    order = sorted(range(len(images)), key=lambda i: images[i].size[0] / images[i].size[1])
    preds = [None] * len(images)
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        chunk = [images[i] for i in idx]
        try:
            out = predictor.predict_batch(chunk)
        except Exception as e:  # fall back to one by one
            print(f'predict_batch failed ({e!r}), predicting one by one')
            out = [predictor.predict(im) for im in chunk]
        for i, p in zip(idx, out):
            preds[i] = p
    return preds


def evaluate_split(config, data_root, split, out_dir, name, batch=64):
    """Predict every crop of `split`, write <name>_metrics.json/_predictions.tsv, return metrics."""
    rows = read_split(data_root, split)
    images = [Image.open(Path(data_root) / rel).convert('RGB') for rel, _, _ in rows]
    predictor = Predictor(config)
    if str(config['device']).startswith('cuda'):
        torch.cuda.synchronize()
    t0 = time.time()
    preds = predict_all(predictor, images, batch)
    if str(config['device']).startswith('cuda'):
        torch.cuda.synchronize()
    seconds = time.time() - t0

    truths = [t for _, t, _ in rows]
    labels = [lab for _, _, lab in rows] if rows and rows[0][2] is not None else None
    result = evaluate(truths, preds, labels)
    result.update({
        'name': name, 'split': split, 'weights': str(config['weights']),
        'device': str(config['device']), 'image_max_width': config['dataset']['image_max_width'],
        'seconds_per_crop': seconds / max(len(rows), 1),
    })
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f'{name}_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    with open(out_dir / f'{name}_predictions.tsv', 'w', encoding='utf-8') as f:
        f.write('image\tlabel\tground_truth\tprediction\tcorrect\n')
        for (rel, t, lab), p in zip(rows, preds):
            f.write(f'{rel}\t{lab or ""}\t{t}\t{p}\t{int(t == p)}\n')
    print(f'== {name} on {split} ({len(rows)} crops, {result["seconds_per_crop"] * 1000:.1f} ms/crop)')
    print(format_table(result))
    return result


def load_config(path, max_width=None, device=None):
    path = Path(path) if path else HERE / 'vgg_transformer.yml'
    config = Cfg.load_config_from_file(str(path))
    if max_width:
        config['dataset']['image_max_width'] = max_width
    config['device'] = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    config['cnn']['pretrained'] = False
    config['predictor']['beamsearch'] = False
    return config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True, help="weights file, or 'pretrained' (official vgg_transformer)")
    ap.add_argument('--config', default=None, help='config yml (default vietocr/vgg_transformer.yml)')
    ap.add_argument('--data-root', default=str(REPO / 'dataset' / 'pres' / 'vietocr'))
    ap.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    ap.add_argument('--max-width', type=int, default=None, help='override image_max_width of the config')
    ap.add_argument('--device', default=None)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--name', default=None)
    ap.add_argument('--out', default=str(REPO / 'results' / 'E-C1'))
    opt = ap.parse_args()

    config = load_config(opt.config, opt.max_width, opt.device)
    config['weights'] = PRETRAINED_URL if opt.weights == 'pretrained' else opt.weights
    name = opt.name or ('C2_pretrained' if opt.weights == 'pretrained' else Path(opt.weights).stem)
    evaluate_split(config, opt.data_root, opt.split, opt.out, name, opt.batch)


if __name__ == '__main__':
    main()
