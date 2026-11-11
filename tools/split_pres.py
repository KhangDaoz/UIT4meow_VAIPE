"""Reproducible train/val/test split of the VAIPE prescriptions (public_train).

The public test set has no labels, so public_train is split into
train / val / test (default 70 / 15 / 15):
  - train: used to fit the model
  - val:   used during training (model selection, early stopping)
  - test:  held out, only used once at the end to report the final score

Reproducibility
  - Input names are sorted before shuffling, so the result does not depend on
    the order the file system lists files in.
  - The shuffle uses random.Random(seed): a private generator, unaffected by any
    other random call.
  - Split sizes: n_val = round(n * val_ratio), n_test = round(n * test_ratio),
    n_train = the rest.
  - The result is written to a manifest JSON (splits/pres_split_seed<seed>.json)
    with the seed, ratios, counts, the file lists and a SHA-256 of the input list.
    The manifest is committed to git; training reads the manifest, so the split
    stays identical even if this code or Python changes.

Usage:
    # make (or re-make) the manifest from the label folder
    python tools/split_pres.py --label-dir <public_train>/prescription/label --seed 42
    # check that an existing manifest matches the data (exit 1 if not)
    python tools/split_pres.py --label-dir <...>/prescription/label --check splits/pres_split_seed42.json
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SEED = 42
DEFAULT_RATIOS = (0.70, 0.15, 0.15)


def list_stems(label_dir):
    return sorted(p.stem for p in Path(label_dir).glob('*.json'))


def names_sha256(stems):
    return hashlib.sha256('\n'.join(sorted(stems)).encode('utf-8')).hexdigest()


def make_split(stems, seed=DEFAULT_SEED, ratios=DEFAULT_RATIOS):
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f'ratios must sum to 1, got {ratios}')
    stems = sorted(stems)
    order = stems[:]
    random.Random(seed).shuffle(order)
    n = len(order)
    n_val = round(n * ratios[1])
    n_test = round(n * ratios[2])
    n_train = n - n_val - n_test
    split = {
        'train': sorted(order[:n_train]),
        'val': sorted(order[n_train:n_train + n_val]),
        'test': sorted(order[n_train + n_val:]),
    }
    return {
        'description': 'VAIPE public_train prescriptions split into train/val/test',
        'seed': seed,
        'ratios': {'train': ratios[0], 'val': ratios[1], 'test': ratios[2]},
        'method': 'sorted stems -> random.Random(seed).shuffle -> '
                  'n_val=round(n*val), n_test=round(n*test), n_train=rest',
        'n_total': n,
        'counts': {k: len(v) for k, v in split.items()},
        'names_sha256': names_sha256(stems),
        'splits': split,
    }


def default_manifest_path(seed):
    return REPO / 'splits' / f'pres_split_seed{seed}.json'


def load_manifest(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def check_manifest(manifest, stems):
    """Return a list of problems (empty = manifest matches the data)."""
    problems = []
    listed = [s for part in manifest['splits'].values() for s in part]
    if len(listed) != len(set(listed)):
        problems.append('some names appear in more than one split')
    missing = sorted(set(listed) - set(stems))
    extra = sorted(set(stems) - set(listed))
    if missing:
        problems.append(f'{len(missing)} names in manifest but not in data, e.g. {missing[:3]}')
    if extra:
        problems.append(f'{len(extra)} names in data but not in manifest, e.g. {extra[:3]}')
    if not missing and not extra and manifest.get('names_sha256') != names_sha256(stems):
        problems.append('names_sha256 does not match')
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label-dir', required=True, help='<public_train>/prescription/label')
    ap.add_argument('--seed', type=int, default=DEFAULT_SEED)
    ap.add_argument('--ratios', type=float, nargs=3, default=list(DEFAULT_RATIOS),
                    metavar=('TRAIN', 'VAL', 'TEST'))
    ap.add_argument('--out', default=None, help='manifest path (default splits/pres_split_seed<seed>.json)')
    ap.add_argument('--check', default=None, help='only check this manifest against the data')
    opt = ap.parse_args()

    stems = list_stems(opt.label_dir)
    if not stems:
        sys.exit(f'No .json labels in {opt.label_dir}')

    if opt.check:
        problems = check_manifest(load_manifest(opt.check), stems)
        if problems:
            sys.exit('Manifest does not match data:\n  ' + '\n  '.join(problems))
        print(f'OK: {opt.check} matches {len(stems)} labels')
        return

    manifest = make_split(stems, opt.seed, tuple(opt.ratios))
    out = Path(opt.out) if opt.out else default_manifest_path(opt.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
        f.write('\n')
    print(f'wrote {out}: {manifest["counts"]} (seed={opt.seed})')


if __name__ == '__main__':
    main()
