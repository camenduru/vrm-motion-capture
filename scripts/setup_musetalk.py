"""Download the MuseTalk v1.5 weights into ``MuseTalk/models``.

Run it with the Python interpreter that MuseTalk uses (it only needs
``huggingface_hub``):

    C:\\Users\\PC\\AppData\\Local\\Programs\\Python\\Python310\\python.exe ^
        scripts\\setup_musetalk.py

Only the files the v1.5 realtime path actually loads are fetched - the
upstream repo also ships v1 checkpoints, training code and syncnet, none of
which inference needs.
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
MUSETALK = ROOT / 'MuseTalk'
MODELS = MUSETALK / 'models'

# (repo, path inside the repo, directory under models/)
FILES = [
    ('TMElyralab/MuseTalk', 'musetalkV15/unet.pth', ''),
    ('TMElyralab/MuseTalk', 'musetalkV15/musetalk.json', ''),
    ('stabilityai/sd-vae-ft-mse', 'config.json', 'sd-vae'),
    ('stabilityai/sd-vae-ft-mse', 'diffusion_pytorch_model.bin', 'sd-vae'),
    ('openai/whisper-tiny', 'config.json', 'whisper'),
    ('openai/whisper-tiny', 'pytorch_model.bin', 'whisper'),
    ('openai/whisper-tiny', 'preprocessor_config.json', 'whisper'),
    ('ManyOtherFunctions/face-parse-bisent', '79999_iter.pth',
     'face-parse-bisent'),
    ('ManyOtherFunctions/face-parse-bisent', 'resnet18-5c106cde.pth',
     'face-parse-bisent'),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force', action='store_true',
                        help='download again even if a file is present')
    args = parser.parse_args()

    MODELS.mkdir(parents=True, exist_ok=True)
    missing = []
    for repo, name, sub in FILES:
        dest = MODELS / sub / name
        if dest.exists() and dest.stat().st_size > 0 and not args.force:
            print(f'[skip] {dest.relative_to(MUSETALK)} '
                  f'({dest.stat().st_size / 1e6:.1f} MB)')
            continue
        print(f'[get ] {repo}/{name}')
        hf_hub_download(repo_id=repo, filename=name,
                        local_dir=str(MODELS / sub))
        if dest.exists():
            print(f'[ ok ] {dest.relative_to(MUSETALK)} '
                  f'({dest.stat().st_size / 1e6:.1f} MB)')
        else:
            missing.append(f'{repo}/{name}')

    if missing:
        print('\nCould not place:')
        for item in missing:
            print('  ' + item)
        return 1

    marker = MUSETALK / '.musetalk_python'
    marker.write_text(sys.executable, encoding='utf-8')
    print(f'[ ok ] MuseTalk interpreter recorded: {sys.executable}')
    print(f'\nMuseTalk weights are in {MODELS}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
