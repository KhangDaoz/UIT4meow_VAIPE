"""Build the YOLO dataset for block B (plan.md 5.2): text boxes on prescriptions, 6 classes.

Classes, in this order: drugname, quantity, usage, diagnose, date, other.
Split: train / val / test = 820 / 176 / 176 (1_vaipe split, see tools/vaipe_data.py).

Output (default dataset/pres/yolov8):
    images/{train,val,test}/<name>.png   symlinks to the original images (copy if symlink fails)
    labels/{train,val,test}/<name>.txt   "class xc yc w h" (normalised), one line per box
    data.yaml                            absolute path, for ultralytics
    stats.json                           images and boxes per class per split

Usage:
    python tools/prepare_yolo_pres.py --data-root 1_vaipe
    python tools/prepare_yolo_pres.py --data-root <public_train>   # split from splits/pres_split_vaipe1.csv
"""
import argparse
import collections
import json
import os
import shutil
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vaipe_data import CLASSES, clip_box, find_data_root, load_records, read_label  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def link_or_copy(src, dst):
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=os.environ.get('DATA_ROOT'),
                    help='1_vaipe folder or public_train (default: search /kaggle/input)')
    ap.add_argument('--manifest', default=None, help='only for public_train (default vaipe1 CSV)')
    ap.add_argument('--out', default=str(REPO / 'dataset' / 'pres' / 'yolov8'))
    opt = ap.parse_args()

    root = opt.data_root or find_data_root()
    if root is None:
        sys.exit('no data found; pass --data-root')
    records, desc = load_records(root, opt.manifest)
    print(f'data: {desc}')
    cls_id = {c: i for i, c in enumerate(CLASSES)}

    out = Path(opt.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    stats = {}
    for split, recs in records.items():
        (out / 'images' / split).mkdir(parents=True)
        (out / 'labels' / split).mkdir(parents=True)
        per_class = collections.Counter()
        n_skip = 0
        unknown = collections.Counter()
        for r in recs:
            with Image.open(r.image) as im:
                w, h = im.size
            lines = []
            for a in read_label(r.label):
                label = a.get('label')
                if label not in cls_id:
                    unknown[label] += 1
                    continue
                x1, y1, x2, y2 = clip_box(a['box'], w, h)
                if x2 - x1 <= 1 or y2 - y1 <= 1:
                    n_skip += 1
                    continue
                lines.append(f'{cls_id[label]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} '
                             f'{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}')
                per_class[label] += 1
            link_or_copy(r.image.resolve(), out / 'images' / split / r.image.name)
            with open(out / 'labels' / split / f'{r.stem}.txt', 'w') as f:
                f.write('\n'.join(lines) + ('\n' if lines else ''))
        stats[split] = {'images': len(recs), 'boxes': sum(per_class.values()),
                        'boxes_per_class': {c: per_class[c] for c in CLASSES},
                        'skipped_degenerate_boxes': n_skip,
                        'skipped_unknown_label': dict(unknown)}
        print(f'{split:5s}: {len(recs)} images, {sum(per_class.values())} boxes '
              f'({", ".join(f"{c}={per_class[c]}" for c in CLASSES)}); '
              f'skipped {n_skip} degenerate, {sum(unknown.values())} unknown label')

    with open(out / 'data.yaml', 'w') as f:
        f.write(f'# made by tools/prepare_yolo_pres.py from: {desc}\n'
                f'path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n')
        f.writelines(f'  {i}: {c}\n' for i, c in enumerate(CLASSES))
    with open(out / 'stats.json', 'w') as f:
        json.dump({'source': desc, **stats}, f, indent=1)
    print(f'dataset ready: {out}/data.yaml')


if __name__ == '__main__':
    main()
