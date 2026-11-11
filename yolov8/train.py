"""Train YOLOv8 to detect drug-name regions on prescription images.

Separate from the YOLOv7 flow: it reuses the YOLO-format data made by
tools/generate_train_detect_pres.py (dataset/pres/yolov7/{images,labels}/{train,val})
and writes its own data.yaml with absolute paths next to it.

Usage (from anywhere):
    python yolov8/train.py --model yolov8s.pt --epochs 100 --batch 16 --device 0
Best weights: <project>/<name>/weights/best.pt  (default yolov8/runs/pres/weights/best.pt)
"""
import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default=str(REPO / 'dataset' / 'pres' / 'yolov7'),
                    help='folder with images/{train,val} and labels/{train,val}')
    ap.add_argument('--model', default='yolov8s.pt', help='pretrained COCO weights (auto-download)')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--device', default='0', help="'0', '0,1' or 'cpu'")
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--patience', type=int, default=30, help='early-stop after N epochs w/o improvement')
    ap.add_argument('--project', default=str(REPO / 'yolov8' / 'runs'))
    ap.add_argument('--name', default='pres')
    opt = ap.parse_args()

    data_dir = Path(opt.data_dir).resolve()
    for split in ('train', 'val'):
        n_img = len(list((data_dir / 'images' / split).glob('*')))
        n_lbl = len(list((data_dir / 'labels' / split).glob('*.txt')))
        if n_img == 0:
            raise SystemExit(f'No images in {data_dir}/images/{split}. '
                             'Run tools/generate_train_detect_pres.py first.')
        print(f'{split}: {n_img} images, {n_lbl} label files')

    # Absolute paths: ultralytics resolves relative paths against its own datasets_dir,
    # not against the current folder.
    data_yaml = data_dir / 'data_yolov8.yaml'
    with open(data_yaml, 'w') as f:
        yaml.safe_dump({'path': str(data_dir), 'train': 'images/train', 'val': 'images/val',
                        'names': {0: 'drugname'}}, f, sort_keys=False)

    model = YOLO(opt.model)
    model.train(data=str(data_yaml), epochs=opt.epochs, batch=opt.batch, imgsz=opt.imgsz,
                device=opt.device, workers=opt.workers, patience=opt.patience,
                project=opt.project, name=opt.name, exist_ok=True, plots=True)


if __name__ == '__main__':
    main()
