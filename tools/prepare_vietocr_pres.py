"""Build the VietOCR dataset (drug-name text lines) from the split manifest.

Same train/val/test split as the YOLOv8 data (splits/pres_split_seed<seed>.json, or the
JSON/CSV given by --manifest), so
no prescription used to test one model was seen by the other while training.

Output (default dataset/pres/vietocr_split):
    img/<prescription>_<k>.jpg   crop of the k-th drug-name box (k in label-file order)
    train.txt / val.txt / test.txt   "img/<file>.jpg<TAB><text>" per line, sorted
    stats.json

Usage:
    python tools/prepare_vietocr_pres.py --data-root <public_train> [--seed 42] [--manifest path]
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_pres import (DEFAULT_SEED, check_manifest, default_manifest_path,  # noqa: E402
                        describe, list_stems, load_manifest)

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=os.environ.get('DATA_ROOT', '/data/public_train'))
    ap.add_argument('--seed', type=int, default=DEFAULT_SEED)
    ap.add_argument('--manifest', default=None, help='default: splits/pres_split_seed<seed>.json')
    ap.add_argument('--out', default=str(REPO / 'dataset' / 'pres' / 'vietocr_split'))
    opt = ap.parse_args()

    pres = Path(opt.data_root) / 'prescription'
    label_dir, image_dir = pres / 'label', pres / 'image'
    stems = list_stems(label_dir)

    manifest_path = Path(opt.manifest) if opt.manifest else default_manifest_path(opt.seed)
    if not manifest_path.exists():
        sys.exit(f'{manifest_path} not found. Run tools/split_pres.py or tools/prepare_yolo_pres.py first.')
    manifest = load_manifest(manifest_path)
    problems = check_manifest(manifest, stems)
    if problems:
        sys.exit(f'{manifest_path} does not match the data:\n  ' + '\n  '.join(problems))
    print(f'using split manifest {manifest_path} ({describe(manifest)})')

    out = Path(opt.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    (out / 'img').mkdir(parents=True)

    stats = {}
    for split, names in manifest['splits'].items():
        lines, n_skip = [], 0
        for name in names:
            img_path = next(iter(sorted(image_dir.glob(name + '.*'))), None)
            if img_path is None:
                sys.exit(f'image for {name} not found in {image_dir}')
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            with open(label_dir / f'{name}.json', encoding='utf-8') as f:
                anns = json.load(f)
            k = 0
            for a in anns:
                if a.get('label') != 'drugname':
                    continue
                text = str(a.get('text', '')).replace('\t', ' ').replace('\n', ' ').strip()
                x1, y1, x2, y2 = (int(round(float(v))) for v in a['box'])
                x1, x2 = sorted((min(max(x1, 0), w), min(max(x2, 0), w)))
                y1, y2 = sorted((min(max(y1, 0), h), min(max(y2, 0), h)))
                if not text or x2 - x1 < 2 or y2 - y1 < 2:
                    n_skip += 1
                    continue
                fname = f'{name}_{k:02d}.jpg'
                cv2.imwrite(str(out / 'img' / fname), img[y1:y2, x1:x2])
                lines.append(f'img/{fname}\t{text}')
                k += 1
        lines.sort()
        with open(out / f'{split}.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        stats[split] = {'prescriptions': len(names), 'lines': len(lines), 'skipped_boxes': n_skip}
        print(f'{split:5s}: {len(names)} prescriptions, {len(lines)} text lines ({n_skip} skipped)')

    with open(out / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=1)
    print(f'dataset ready: {out}')


if __name__ == '__main__':
    main()
