"""Build the VietOCR dataset for block C (plan.md 5.3): every labelled text box, cropped.

Each box (all 6 classes by default) is cropped from the original image with a 2 px border,
using the ground-truth box. Split: train / val / test = 820 / 176 / 176 (tools/vaipe_data.py),
so the test crops are the "crops by GT box" that experiment E-C1 scores OCR engines on.

Output (default dataset/pres/vietocr):
    img/{train,val,test}/<image>_<k>.png    crop of the k-th box of the label file
    {train,val,test}.txt                    "img/<split>/<file>.png<TAB><text>" (VietOCR format)
    {train,val,test}_meta.tsv               path, label (class), text  -> per-class CER
    stats.json

Text is normalised to Unicode NFC with single spaces (tools/vaipe_data.clean_text).
Boxes whose text has a character outside the VietOCR vocab are skipped (and counted).

Usage:
    python tools/prepare_vietocr_pres.py --data-root 1_vaipe
    python tools/prepare_vietocr_pres.py --data-root <public_train> --classes drugname quantity
"""
import argparse
import collections
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vaipe_data import (CLASSES, clean_text, clip_box, find_data_root,  # noqa: E402
                        load_records, read_label)

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=os.environ.get('DATA_ROOT'),
                    help='1_vaipe folder or public_train (default: search /kaggle/input)')
    ap.add_argument('--manifest', default=None, help='only for public_train (default vaipe1 CSV)')
    ap.add_argument('--classes', nargs='+', default=CLASSES, choices=CLASSES)
    ap.add_argument('--pad', type=int, default=2, help='border in px around each box')
    ap.add_argument('--vocab-config', default=str(REPO / 'vietocr' / 'vgg_transformer.yml'))
    ap.add_argument('--out', default=str(REPO / 'dataset' / 'pres' / 'vietocr'))
    opt = ap.parse_args()

    root = opt.data_root or find_data_root()
    if root is None:
        sys.exit('no data found; pass --data-root')
    records, desc = load_records(root, opt.manifest)
    print(f'data: {desc}')
    with open(opt.vocab_config, encoding='utf-8') as f:
        vocab = set(yaml.safe_load(f)['vocab'])
    keep = set(opt.classes)

    out = Path(opt.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    stats = {}
    for split, recs in records.items():
        (out / 'img' / split).mkdir(parents=True)
        lines, meta = [], []
        per_class = collections.Counter()
        skip = collections.Counter()
        bad_chars = collections.Counter()
        for r in recs:
            img = cv2.imread(str(r.image))
            if img is None:
                sys.exit(f'cannot read {r.image}')
            h, w = img.shape[:2]
            for k, a in enumerate(read_label(r.label)):
                label = a.get('label')
                if label not in keep:
                    continue
                text = clean_text(a.get('text', ''))
                x1, y1, x2, y2 = clip_box(a['box'], w, h)
                if not text:
                    skip['empty_text'] += 1
                    continue
                if x2 - x1 < 2 or y2 - y1 < 2:
                    skip['degenerate_box'] += 1
                    continue
                oov = [c for c in text if c not in vocab]
                if oov:
                    bad_chars.update(oov)
                    skip['char_not_in_vocab'] += 1
                    continue
                x1, y1 = max(x1 - opt.pad, 0), max(y1 - opt.pad, 0)
                x2, y2 = min(x2 + opt.pad, w), min(y2 + opt.pad, h)
                rel = f'img/{split}/{r.stem}_{k:03d}.png'
                cv2.imwrite(str(out / rel), img[y1:y2, x1:x2])
                lines.append(f'{rel}\t{text}')
                meta.append(f'{rel}\t{label}\t{text}')
                per_class[label] += 1
        lines.sort()
        meta.sort()
        with open(out / f'{split}.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        with open(out / f'{split}_meta.tsv', 'w', encoding='utf-8') as f:
            f.write('path\tlabel\ttext\n' + '\n'.join(meta) + '\n')
        stats[split] = {'images': len(recs), 'crops': len(lines),
                        'crops_per_class': {c: per_class[c] for c in opt.classes},
                        'skipped': dict(skip), 'chars_not_in_vocab': dict(bad_chars)}
        print(f'{split:5s}: {len(recs)} images, {len(lines)} crops '
              f'({", ".join(f"{c}={per_class[c]}" for c in opt.classes)}); skipped {dict(skip)}')

    with open(out / 'stats.json', 'w', encoding='utf-8') as f:
        json.dump({'source': desc, 'classes': opt.classes, 'pad': opt.pad, **stats},
                  f, indent=1, ensure_ascii=False)
    print(f'dataset ready: {out}')


if __name__ == '__main__':
    main()
