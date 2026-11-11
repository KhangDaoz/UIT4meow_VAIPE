"""Train YOLOv8 to find and classify text boxes on prescriptions (plan.md 5.2, block B), 6 classes.

Classes: drugname, quantity, usage, diagnose, date, other.
Data: dataset/pres/yolov8/data.yaml, made by tools/prepare_yolo_pres.py.
  - train: fit the model
  - val:   pick best.pt, early stopping, and the conf threshold that maximises mean F1
  - test:  scored once at the end with best.pt -> <run>/test_metrics.json (E-B2 numbers)

Defaults follow the plan: yolov8s, imgsz 1024, 50 epochs, batch 16, patience 15, seed 42,
fliplr 0 (flipping breaks text), mosaic 0.5.

Usage:
    python yolov8/train.py --device 0          # one T4
    python yolov8/train.py --device 0,1        # two T4 (DDP), batch is split over the GPUs
Best weights: <project>/<name>/weights/best.pt (default yolov8/runs/vaipe6/weights/best.pt)
"""
import argparse
import json
from pathlib import Path

import yaml
from ultralytics import YOLO

REPO = Path(__file__).resolve().parent.parent


def summarise(metrics, names):
    """Overall and per-class P / R / F1 / mAP50 / mAP50-95 from an ultralytics val() result."""
    box = metrics.box
    p, r = float(box.mp), float(box.mr)
    out = {'precision': p, 'recall': r, 'f1': 2 * p * r / (p + r) if p + r else 0.0,
           'mAP50': float(box.map50), 'mAP50-95': float(box.map), 'per_class': {}}
    for i, c in enumerate(box.ap_class_index):
        cp, cr, ap50, ap = (float(v) for v in box.class_result(i))
        out['per_class'][names[int(c)]] = {
            'precision': cp, 'recall': cr, 'f1': 2 * cp * cr / (cp + cr) if cp + cr else 0.0,
            'mAP50': ap50, 'mAP50-95': ap}
    return out


def best_conf(metrics):
    """Confidence that maximises the mean F1 over classes (from the F1 curve), or None."""
    try:
        f1 = metrics.box.f1_curve.mean(0)
        i = int(f1.argmax())
        return float(metrics.box.px[i]), float(f1[i])
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(REPO / 'dataset' / 'pres' / 'yolov8' / 'data.yaml'),
                    help='data.yaml made by tools/prepare_yolo_pres.py')
    ap.add_argument('--model', default='yolov8s.pt', help='pretrained COCO weights (auto-download)')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--imgsz', type=int, default=1024)
    ap.add_argument('--device', default='0', help="'0', '0,1' or 'cpu'")
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--patience', type=int, default=15, help='early-stop after N epochs w/o improvement')
    ap.add_argument('--fliplr', type=float, default=0.0, help='0: flipping makes text unreadable')
    ap.add_argument('--mosaic', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--fraction', type=float, default=1.0, help='share of train images used (E-B3)')
    ap.add_argument('--project', default=str(REPO / 'yolov8' / 'runs'))
    ap.add_argument('--name', default='vaipe6')
    opt = ap.parse_args()

    data_yaml = Path(opt.data).resolve()
    if not data_yaml.exists():
        raise SystemExit(f'{data_yaml} not found. Run tools/prepare_yolo_pres.py first.')
    with open(data_yaml) as f:
        data = yaml.safe_load(f)
    names = data['names']
    root = Path(data['path'])
    for split in ('train', 'val', 'test'):
        n = len(list((root / 'images' / split).glob('*')))
        if n == 0:
            raise SystemExit(f'No images in {root}/images/{split}.')
        print(f'{split}: {n} images')
    print(f'classes: {names}')

    model = YOLO(opt.model)
    model.train(data=str(data_yaml), epochs=opt.epochs, batch=opt.batch, imgsz=opt.imgsz,
                device=opt.device, workers=opt.workers, patience=opt.patience,
                fliplr=opt.fliplr, mosaic=opt.mosaic, fraction=opt.fraction,
                seed=opt.seed, deterministic=True,
                project=opt.project, name=opt.name, exist_ok=True, plots=True)

    # ---- val (conf threshold) and test (final numbers) with best.pt, one GPU ----
    run_dir = Path(opt.project) / opt.name
    best = YOLO(str(run_dir / 'weights' / 'best.pt'))
    dev = opt.device.split(',')[0]
    kw = dict(data=str(data_yaml), imgsz=opt.imgsz, batch=opt.batch, device=dev,
              project=opt.project, exist_ok=True, plots=True)
    val = best.val(split='val', name=opt.name + '_val', **kw)
    conf, f1 = best_conf(val)
    test = best.val(split='test', name=opt.name + '_test', **kw)

    result = {
        'split': 'test',
        'weights': 'best.pt (selected on val)',
        **summarise(test, names),
        'val': summarise(val, names),
        'conf_max_f1_on_val': conf,
        'mean_f1_at_that_conf_on_val': f1,
        'params': {'model': opt.model, 'imgsz': opt.imgsz, 'epochs': opt.epochs,
                   'batch': opt.batch, 'patience': opt.patience, 'fliplr': opt.fliplr,
                   'mosaic': opt.mosaic, 'seed': opt.seed, 'fraction': opt.fraction},
    }
    with open(run_dir / 'test_metrics.json', 'w') as f:
        json.dump(result, f, indent=1)

    print(f'conf that maximises mean F1 on val: {conf}')
    print(f'{"class":10s} {"P":>7s} {"R":>7s} {"F1":>7s} {"mAP50":>7s} {"mAP50-95":>8s}')
    rows = [('all', result)] + list(result['per_class'].items())
    for c, m in rows:
        print(f'{c:10s} {m["precision"]:7.4f} {m["recall"]:7.4f} {m["f1"]:7.4f} '
              f'{m["mAP50"]:7.4f} {m["mAP50-95"]:8.4f}')
    print(f'TEST RESULT (YOLOv8): P={result["precision"]:.4f}  R={result["recall"]:.4f}  '
          f'F1={result["f1"]:.4f}  mAP50={result["mAP50"]:.4f}  mAP50-95={result["mAP50-95"]:.4f}')


if __name__ == '__main__':
    main()
