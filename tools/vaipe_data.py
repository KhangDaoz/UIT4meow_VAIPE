"""Find the VAIPE prescription data and list the train/val/test images with their labels.

Two layouts are accepted. Both give the same split (train/val/test = 820/176/176, grouped by
patient, see 1_vaipe/README.md):

1. 1_vaipe (the team's split, plan.md section 2.1):
       <root>/{train,valid,test}/images/*.png
       <root>/{train,valid,test}/labels/*.json
   The split is the folder (valid -> val). test_unlabeled/ is ignored.

2. Original Kaggle dataset (vaipepill2022), folder public_train:
       <root>/prescription/image/*.png
       <root>/prescription/label/*.json
   The split is read from a manifest, default splits/pres_split_vaipe1.csv
   (a copy of 1_vaipe/split_manifest.csv).

Label file: a list of {"id", "text", "label", "box": [xmin, ymin, xmax, ymax], "mapping"?}.

Usage:
    python tools/vaipe_data.py --find                # print the data root found under /kaggle/input
    python tools/vaipe_data.py --data-root 1_vaipe   # print the split and box counts per class
"""
import argparse
import collections
import json
import os
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_pres import check_manifest, describe, list_stems, load_manifest  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
CLASSES = ['drugname', 'quantity', 'usage', 'diagnose', 'date', 'other']   # YOLO class order
SPLITS = ('train', 'val', 'test')
FOLDERS = {'train': 'train', 'val': 'valid', 'test': 'test'}                 # 1_vaipe folder names
DEFAULT_MANIFEST = REPO / 'splits' / 'pres_split_vaipe1.csv'
IMAGE_EXTS = ('.png', '.jpg', '.jpeg')

Record = collections.namedtuple('Record', 'stem image label')


def layout(root):
    """'vaipe1', 'public' or None."""
    root = Path(root)
    if all((root / FOLDERS[s] / 'labels').is_dir() for s in SPLITS):
        return 'vaipe1'
    if (root / 'prescription' / 'label').is_dir():
        return 'public'
    return None


def find_data_root(search=('/kaggle/input',), max_depth=6):
    """First 1_vaipe-style folder under `search`, else the first public_train folder."""
    found = {'vaipe1': [], 'public': []}
    skip = {'image', 'images', 'label', 'labels', 'pill'}
    queue = [(Path(s), 0) for s in search if Path(s).is_dir()]
    while queue:
        d, depth = queue.pop(0)
        kind = layout(d)
        if kind == 'vaipe1' or (kind == 'public' and d.name == 'public_train'):
            found[kind].append(d)
            continue
        if depth < max_depth:
            try:
                subs = sorted(e.path for e in os.scandir(d) if e.is_dir() and e.name not in skip)
            except OSError:
                continue
            queue += [(Path(p), depth + 1) for p in subs]
    hits = found['vaipe1'] or found['public']
    return hits[0] if hits else None


def _image_for(image_dir, stem):
    for ext in IMAGE_EXTS:
        p = image_dir / (stem + ext)
        if p.exists():
            return p
    return None


def load_records(root, manifest=None):
    """Return ({'train': [Record], 'val': [...], 'test': [...]}, description)."""
    root = Path(root)
    kind = layout(root)
    records = {}
    if kind == 'vaipe1':
        for split in SPLITS:
            label_dir = root / FOLDERS[split] / 'labels'
            image_dir = root / FOLDERS[split] / 'images'
            records[split] = [Record(s, _image_for(image_dir, s), label_dir / f'{s}.json')
                              for s in list_stems(label_dir)]
        desc = f'1_vaipe folders in {root}'
    elif kind == 'public':
        label_dir = root / 'prescription' / 'label'
        image_dir = root / 'prescription' / 'image'
        manifest_path = Path(manifest) if manifest else DEFAULT_MANIFEST
        m = load_manifest(manifest_path)
        problems = check_manifest(m, list_stems(label_dir))
        if problems:
            sys.exit(f'{manifest_path} does not match {label_dir}:\n  ' + '\n  '.join(problems))
        for split in SPLITS:
            records[split] = [Record(s, _image_for(image_dir, s), label_dir / f'{s}.json')
                              for s in m['splits'][split]]
        desc = f'{root} split by {manifest_path.name} ({describe(m)})'
    else:
        sys.exit(f'{root}: not a 1_vaipe folder ({{train,valid,test}}/labels) '
                 f'nor public_train (prescription/label)')
    for split, recs in records.items():
        missing = [r.stem for r in recs if r.image is None]
        if missing:
            sys.exit(f'{split}: {len(missing)} labels without image, e.g. {missing[:3]}')
        if not recs:
            sys.exit(f'{split}: no images')
    return records, desc


def clean_text(text):
    """Unicode NFC, one space between words (same normalisation as the OCR metrics)."""
    return ' '.join(unicodedata.normalize('NFC', str(text)).split())


def read_label(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def clip_box(box, w, h):
    """[xmin, ymin, xmax, ymax] clipped to the image, as ints."""
    x1, y1, x2, y2 = (float(v) for v in box)
    x1, x2 = sorted((min(max(x1, 0), w), min(max(x2, 0), w)))
    y1, y2 = sorted((min(max(y1, 0), h), min(max(y2, 0), h)))
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--manifest', default=None, help='only for public_train (default vaipe1 CSV)')
    ap.add_argument('--find', action='store_true', help='print the data root under /kaggle/input')
    opt = ap.parse_args()

    if opt.find:
        root = find_data_root()
        if root is None:
            sys.exit(1)
        print(root)
        return
    root = opt.data_root or find_data_root()
    if root is None:
        sys.exit('no data found; pass --data-root')
    records, desc = load_records(root, opt.manifest)
    print(desc)
    for split, recs in records.items():
        count = collections.Counter(a['label'] for r in recs for a in read_label(r.label))
        print(f'{split:5s}: {len(recs)} images, ' + ', '.join(f'{c}={count[c]}' for c in CLASSES))


if __name__ == '__main__':
    main()
