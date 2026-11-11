"""Train VietOCR (vgg_transformer) on the 70/15/15 split, then score it on test.

Data: dataset/pres/vietocr_split (made by tools/prepare_vietocr_pres.py).
  - train: fit the model (fine-tuned from the official vgg_transformer weights)
  - val:   checked every --valid-every iterations; the best one is exported
  - test:  scored once at the end with the exported weights -> <work-dir>/test_metrics.json

The old flow (vietocr/train_vietocr.py, 80/20 split) is unchanged.

Usage:
    python vietocr/train_vietocr_split.py --iters 20000 --device cuda:0 --seed 42
Output (in --work-dir, default vietocr/runs_split):
    weights/transformerocr.pth, test_metrics.json, test_predictions.tsv, config.yml, train.log
"""
import argparse
import json
import os
import random
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import torch

# vietocr imports imgaug at module level, but training only uses the albumentations
# augmenter (ImgAugTransformV2). imgaug 0.4.0 fails to import on NumPy 2, so stub it.
try:
    import imgaug  # noqa: F401
except Exception:
    _stub = types.ModuleType('imgaug')
    _stub.augmenters = types.ModuleType('imgaug.augmenters')
    sys.modules['imgaug'] = _stub
    sys.modules['imgaug.augmenters'] = _stub.augmenters

from PIL import Image  # noqa: E402
from vietocr.model.trainer import Trainer  # noqa: E402
from vietocr.tool.config import Cfg  # noqa: E402
from vietocr.tool.predictor import Predictor  # noqa: E402
from vietocr.tool.utils import compute_accuracy  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def read_annotation(path):
    pairs = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line:
                rel, text = line.split('\t', 1)
                pairs.append((rel, text))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=str(REPO / 'dataset' / 'pres' / 'vietocr_split'))
    ap.add_argument('--work-dir', default=str(HERE / 'runs_split'))
    ap.add_argument('--iters', type=int, default=20000)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--valid-every', type=int, default=3000)
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--num-workers', type=int, default=3)
    opt = ap.parse_args()

    data_root = Path(opt.data_root).resolve()
    for split in ('train', 'val', 'test'):
        if not (data_root / f'{split}.txt').exists():
            raise SystemExit(f'{data_root}/{split}.txt not found. Run tools/prepare_vietocr_pres.py first.')
        print(f'{split}: {len(read_annotation(data_root / f"{split}.txt"))} lines')

    # Work in a clean folder: VietOCR caches its lmdb datasets in the current folder
    # and silently reuses them if they exist.
    work = Path(opt.work_dir).resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    os.chdir(work)

    set_seed(opt.seed)

    local_cfg = HERE / 'vgg_transformer.yml'
    config = Cfg.load_config_from_file(str(local_cfg)) if local_cfg.exists() \
        else Cfg.load_config_from_name('vgg_transformer')
    config['device'] = opt.device
    config['cnn']['pretrained'] = False  # all weights come from config['pretrain'] below
    config['dataset'].update({
        'name': 'pres_split',
        'data_root': str(data_root),
        'train_annotation': 'train.txt',
        'valid_annotation': 'val.txt',
    })
    config['trainer'].update({
        'iters': opt.iters,
        'batch_size': opt.batch_size,
        'print_every': min(200, opt.iters),
        'valid_every': min(opt.valid_every, opt.iters),  # weights are exported only here
        'metrics': 100000,                                # use the whole val set
        'export': str(work / 'weights' / 'transformerocr.pth'),
        'checkpoint': str(work / 'checkpoint' / 'transformerocr_checkpoint.pth'),
        'log': str(work / 'train.log'),
    })
    config['dataloader']['num_workers'] = opt.num_workers
    config['seed'] = opt.seed

    trainer = Trainer(config, pretrained=True)
    trainer.config.save(str(work / 'config.yml'))
    trainer.train()

    # ---- final score on the held-out test split, with the exported (best on val) weights ----
    weights = Path(config['trainer']['export'])
    if not weights.exists():
        raise SystemExit('No weights were exported (val accuracy never improved above 0).')
    config['weights'] = str(weights)
    predictor = Predictor(config)

    pairs = read_annotation(data_root / 'test.txt')
    preds = []
    for rel, _ in pairs:
        preds.append(predictor.predict(Image.open(data_root / rel).convert('RGB')))
    truths = [t for _, t in pairs]
    result = {
        'split': 'test',
        'weights': 'transformerocr.pth (best on val)',
        'lines': len(pairs),
        'acc_full_sequence': float(compute_accuracy(truths, preds, mode='full_sequence')),
        'acc_per_char': float(compute_accuracy(truths, preds, mode='per_char')),
        'seed': opt.seed,
        'iters': opt.iters,
    }
    with open(work / 'test_metrics.json', 'w') as f:
        json.dump(result, f, indent=1)
    with open(work / 'test_predictions.tsv', 'w', encoding='utf-8') as f:
        f.write('image\tground_truth\tprediction\tcorrect\n')
        for (rel, t), p in zip(pairs, preds):
            f.write(f'{rel}\t{t}\t{p}\t{int(t == p)}\n')
    print(f'TEST RESULT (VietOCR): lines={result["lines"]}  '
          f'acc_full_sequence={result["acc_full_sequence"]:.4f}  acc_per_char={result["acc_per_char"]:.4f}')


if __name__ == '__main__':
    main()
