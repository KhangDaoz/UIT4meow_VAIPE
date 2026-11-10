import os
import sys
import types

# vietocr imports imgaug at module level, but training only uses the albumentations
# augmenter (ImgAugTransformV2). imgaug 0.4.0 fails to import on NumPy 2, so stub it.
try:
    import imgaug  # noqa: F401
except Exception:
    _stub = types.ModuleType('imgaug')
    _stub.augmenters = types.ModuleType('imgaug.augmenters')
    sys.modules['imgaug'] = _stub
    sys.modules['imgaug.augmenters'] = _stub.augmenters

from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer

HERE = os.path.dirname(os.path.abspath(__file__))
local_cfg = os.path.join(HERE, 'vgg_transformer.yml')
if os.path.exists(local_cfg):
    config = Cfg.load_config_from_file(local_cfg)
else:
    config = Cfg.load_config_from_name('vgg_transformer')

dataset_params = {
    'name': 'hw',
    'data_root': '../dataset/pres/vietocr',
    'train_annotation': 'train.txt',
    'valid_annotation': 'val.txt'
}

iters = int(os.environ.get('OCR_ITERS', 20000))
params = {
    'print_every': min(200, iters),
    'valid_every': min(15 * 200, iters),  # weights are exported only at validation steps
    'iters': iters,
    'export': './weights/transformerocr.pth',
    'metrics': 2000
}

config['trainer'].update(params)
config['dataset'].update(dataset_params)
config['device'] = 'cuda:0'
trainer = Trainer(config, pretrained=True)
trainer.config.save('config.yml')
trainer.train()
