"""Train YOLOv8 to detect drug-name regions on prescription images, then score it on test.

Data: dataset/pres/yolov8/data.yaml, made by tools/prepare_yolo_pres.py from the
train/val/test split manifest (splits/pres_split_seed<seed>.json).
  - train: fit the model
  - val:   pick the best epoch (best.pt) and early stopping
  - test:  scored once at the end with best.pt -> <run>/test_metrics.json

Usage (from anywhere):
    python yolov8/train.py --model yolov8s.pt --epochs 100 --batch 16 --device 0 --seed 42
Best weights: <project>/<name>/weights/best.pt  (default yolov8/runs/pres/weights/best.pt)
"""
import argparse
import json
from pathlib import Path

import yaml
from ultralytics import YOLO

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(REPO / 'dataset' / 'pres' / 'yolov8' / 'data.yaml'),
                    help='data.yaml made by tools/prepare_yolo_pres.py')
    ap.add_argument('--model', default='yolov8s.pt', help='pretrained COCO weights (auto-download)')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--device', default='0', help="'0', '0,1' or 'cpu'")
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--patience', type=int, default=30, help='early-stop after N epochs w/o improvement')
    ap.add_argument('--seed', type=int, default=42, help='training seed (augmentation, init, loader)')
    ap.add_argument('--project', default=str(REPO / 'yolov8' / 'runs'))
    ap.add_argument('--name', default='pres')
    opt = ap.parse_args()

    data_yaml = Path(opt.data).resolve()
    if not data_yaml.exists():
        raise SystemExit(f'{data_yaml} not found. Run tools/prepare_yolo_pres.py first.')
    with open(data_yaml) as f:
        data = yaml.safe_load(f)
    root = Path(data['path'])
    for split in ('train', 'val', 'test'):
        n = len(list((root / 'images' / split).glob('*')))
        if n == 0:
            raise SystemExit(f'No images in {root}/images/{split}.')
        print(f'{split}: {n} images')

    model = YOLO(opt.model)
    model.train(data=str(data_yaml), epochs=opt.epochs, batch=opt.batch, imgsz=opt.imgsz,
                device=opt.device, workers=opt.workers, patience=opt.patience,
                seed=opt.seed, deterministic=True,
                project=opt.project, name=opt.name, exist_ok=True, plots=True)

    # ---- final score on the held-out test split, with the best (by val) weights ----
    run_dir = Path(opt.project) / opt.name
    best = YOLO(str(run_dir / 'weights' / 'best.pt'))
    metrics = best.val(data=str(data_yaml), split='test', imgsz=opt.imgsz, batch=opt.batch,
                       device=opt.device.split(',')[0], project=opt.project,
                       name=opt.name + '_test', exist_ok=True, plots=True)
    result = {
        'split': 'test',
        'weights': 'best.pt (selected on val)',
        'precision': float(metrics.box.mp),
        'recall': float(metrics.box.mr),
        'mAP50': float(metrics.box.map50),
        'mAP50-95': float(metrics.box.map),
        'seed': opt.seed,
        'model': opt.model,
        'imgsz': opt.imgsz,
    }
    with open(run_dir / 'test_metrics.json', 'w') as f:
        json.dump(result, f, indent=1)
    print('TEST RESULT: ' + '  '.join(f'{k}={v:.4f}' for k, v in result.items()
                                      if isinstance(v, float)))


if __name__ == '__main__':
    main()
