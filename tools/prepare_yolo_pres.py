"""Build the YOLO dataset (drug-name boxes on prescriptions) from a split manifest.

Output (default dataset/pres/yolov8):
    images/{train,val,test}/<name>.png   symlinks to the original images (copy if symlink fails)
    labels/{train,val,test}/<name>.txt   one line per drug-name box: "0 xc yc w h" (normalised)
    data.yaml                            absolute paths, for ultralytics
    split_manifest.json                  copy of the manifest used

The split comes from splits/pres_split_seed<seed>.json (made by tools/split_pres.py).
If that file does not exist it is created from the label folder with the same seed,
so the split is always the same for the same seed and data.

Usage:
    python tools/prepare_yolo_pres.py --data-root <public_train> [--seed 42] [--manifest path]
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_pres import (DEFAULT_SEED, check_manifest, default_manifest_path,  # noqa: E402
                        list_stems, load_manifest, make_split)

REPO = Path(__file__).resolve().parent.parent


def to_yolo_line(box, img_w, img_h):
    x1, y1, x2, y2 = (float(v) for v in box)
    x1, x2 = sorted((min(max(x1, 0), img_w), min(max(x2, 0), img_w)))
    y1, y2 = sorted((min(max(y1, 0), img_h), min(max(y2, 0), img_h)))
    w, h = x2 - x1, y2 - y1
    if w <= 1 or h <= 1:
        return None
    return f'0 {(x1 + x2) / 2 / img_w:.6f} {(y1 + y2) / 2 / img_h:.6f} {w / img_w:.6f} {h / img_h:.6f}'


def link_or_copy(src, dst):
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=os.environ.get('DATA_ROOT', '/data/public_train'),
                    help='public_train folder (has prescription/image and prescription/label)')
    ap.add_argument('--seed', type=int, default=DEFAULT_SEED)
    ap.add_argument('--manifest', default=None, help='default: splits/pres_split_seed<seed>.json')
    ap.add_argument('--out', default=str(REPO / 'dataset' / 'pres' / 'yolov8'))
    opt = ap.parse_args()

    pres = Path(opt.data_root) / 'prescription'
    label_dir, image_dir = pres / 'label', pres / 'image'
    stems = list_stems(label_dir)
    if not stems:
        sys.exit(f'No labels in {label_dir}')

    manifest_path = Path(opt.manifest) if opt.manifest else default_manifest_path(opt.seed)
    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
        problems = check_manifest(manifest, stems)
        if problems:
            sys.exit(f'{manifest_path} does not match the data:\n  ' + '\n  '.join(problems))
        print(f'using split manifest {manifest_path} (seed={manifest["seed"]})')
    else:
        manifest = make_split(stems, opt.seed)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=1, ensure_ascii=False)
            f.write('\n')
        print(f'created split manifest {manifest_path} (seed={opt.seed})')

    out = Path(opt.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    stats = {}
    for split, names in manifest['splits'].items():
        (out / 'images' / split).mkdir(parents=True)
        (out / 'labels' / split).mkdir(parents=True)
        n_box = n_skip = n_empty = 0
        for name in names:
            img_path = next(iter(sorted(image_dir.glob(name + '.*'))), None)
            if img_path is None:
                sys.exit(f'image for {name} not found in {image_dir}')
            with Image.open(img_path) as im:
                img_w, img_h = im.size
            with open(label_dir / f'{name}.json', encoding='utf-8') as f:
                anns = json.load(f)
            lines = []
            for a in anns:
                if a.get('label') != 'drugname':
                    continue
                line = to_yolo_line(a['box'], img_w, img_h)
                if line is None:
                    n_skip += 1
                else:
                    lines.append(line)
            n_box += len(lines)
            n_empty += not lines
            link_or_copy(img_path.resolve(), out / 'images' / split / img_path.name)
            with open(out / 'labels' / split / f'{name}.txt', 'w') as f:
                f.write('\n'.join(lines) + ('\n' if lines else ''))
        stats[split] = {'images': len(names), 'boxes': n_box, 'images_without_box': n_empty,
                        'skipped_degenerate_boxes': n_skip}
        print(f'{split:5s}: {len(names)} images, {n_box} drug-name boxes'
              f' ({n_empty} images without box, {n_skip} degenerate boxes skipped)')

    with open(out / 'data.yaml', 'w') as f:
        f.write(f'# made by tools/prepare_yolo_pres.py from {manifest_path.name}\n'
                f'path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\n'
                f'names:\n  0: drugname\n')
    shutil.copy2(manifest_path, out / 'split_manifest.json')
    with open(out / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=1)
    print(f'dataset ready: {out}/data.yaml')


if __name__ == '__main__':
    main()
