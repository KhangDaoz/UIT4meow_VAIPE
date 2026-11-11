"""Fine-tune VietOCR (vgg_transformer) on all text boxes of the prescriptions = plan.md option C3.

Data: dataset/pres/vietocr (made by tools/prepare_vietocr_pres.py), crops by GT box of all
6 classes, split train / val / test = 820 / 176 / 176 images.
  - train: fit the model, starting from the official vgg_transformer weights
  - val:   checked every --valid-every iterations; the best (exact-match accuracy) is exported
  - test:  scored once at the end -> CER / WER / accuracy, overall and per class

Before training, the official pretrained model (option C2) is scored on the same test crops,
so the E-C1 table gets C2 and C3 from one run (skip with --skip-pretrained-eval).

Usage:
    python vietocr/train_vietocr_split.py --iters 20000 --device cuda:0 --seed 42
Output (in --work-dir, default vietocr/runs_vaipe):
    weights/transformerocr.pth          fine-tuned weights (best on val)
    config.yml                          config to load them (image_max_width etc.)
    C3_finetune_metrics.json            test score of the fine-tuned model
    C3_finetune_predictions.tsv         every test crop: class, ground truth, prediction
    C2_pretrained_metrics.json / _predictions.tsv   same for the pretrained model
    train.log
"""
import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from eval_vietocr import PRETRAINED_URL, evaluate_split, read_split  # also stubs imgaug

from vietocr.model.trainer import Trainer  # noqa: E402
from vietocr.tool.config import Cfg  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def main():
    # VietOCR's dataset holds an lmdb handle that cannot be pickled, so DataLoader workers must
    # be forked (Python 3.14 changed the Linux default to forkserver).
    if sys.platform.startswith('linux'):
        torch.multiprocessing.set_start_method('fork', force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=str(REPO / 'dataset' / 'pres' / 'vietocr'))
    ap.add_argument('--work-dir', default=str(HERE / 'runs_vaipe'))
    ap.add_argument('--iters', type=int, default=20000)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--valid-every', type=int, default=2000)
    ap.add_argument('--max-width', type=int, default=1024,
                    help='image_max_width at height 32; ~5%% of the crops are wider than 512')
    ap.add_argument('--lr', type=float, default=3e-4, help='max lr of the one-cycle schedule')
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--num-workers', type=int, default=3)
    ap.add_argument('--skip-pretrained-eval', action='store_true')
    opt = ap.parse_args()

    data_root = Path(opt.data_root).resolve()
    for split in ('train', 'val', 'test'):
        if not (data_root / f'{split}.txt').exists():
            raise SystemExit(f'{data_root}/{split}.txt not found. Run tools/prepare_vietocr_pres.py first.')
        print(f'{split}: {len(read_split(data_root, split))} crops')

    # Work in a clean folder: VietOCR caches its lmdb datasets in the current folder
    # and silently reuses them if they exist.
    work = Path(opt.work_dir).resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    os.chdir(work)
    set_seed(opt.seed)

    config = Cfg.load_config_from_file(str(HERE / 'vgg_transformer.yml'))
    config['device'] = opt.device
    config['cnn']['pretrained'] = False  # all weights come from config['pretrain']
    config['dataset'].update({
        'name': 'pres_vaipe',
        'data_root': str(data_root),
        'train_annotation': 'train.txt',
        'valid_annotation': 'val.txt',
        'image_max_width': opt.max_width,
    })
    config['trainer'].update({
        'iters': opt.iters,
        'batch_size': opt.batch_size,
        'print_every': min(200, opt.iters),
        'valid_every': min(opt.valid_every, opt.iters),   # weights are exported only here
        'metrics': 100000,                                 # use the whole val set
        'export': str(work / 'weights' / 'transformerocr.pth'),
        'checkpoint': str(work / 'checkpoint' / 'transformerocr_checkpoint.pth'),
        'log': str(work / 'train.log'),
    })
    config['optimizer']['max_lr'] = opt.lr
    config['dataloader']['num_workers'] = opt.num_workers
    config['seed'] = opt.seed

    # ---- C2: pretrained model, no fine-tuning, on the test crops ----
    if not opt.skip_pretrained_eval:
        c2 = Cfg(dict(config))
        c2['weights'] = PRETRAINED_URL
        evaluate_split(c2, data_root, 'test', work, 'C2_pretrained')

    # ---- C3: fine-tune ----
    (work / 'weights').mkdir(exist_ok=True)
    (work / 'checkpoint').mkdir(exist_ok=True)
    trainer = Trainer(config, pretrained=True)
    trainer.train()

    weights = Path(config['trainer']['export'])
    if not weights.exists():
        raise SystemExit('No weights were exported (val accuracy never improved above 0).')
    config['weights'] = str(weights)
    config.save(str(work / 'config.yml'))
    result = evaluate_split(config, data_root, 'test', work, 'C3_finetune')
    result.update({'seed': opt.seed, 'iters': opt.iters, 'batch_size': opt.batch_size,
                   'max_lr': opt.lr, 'selected_on': 'val exact-match accuracy'})
    with open(work / 'C3_finetune_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    a = result['all']
    print(f'TEST RESULT (VietOCR C3): crops={a["lines"]}  CER={a["cer"]:.4f}  WER={a["wer"]:.4f}  '
          f'acc={a["acc_full_sequence"]:.4f}')


if __name__ == '__main__':
    main()
