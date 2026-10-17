"""Find screenshots that show the lab's browser title/address bar and crop it away.

python scripts/crop_lab_frame.py            # list suspicious screenshots
python scripts/crop_lab_frame.py --apply    # crop them (originals copied to ~/screenshot_backup)

The lab's remote-desktop link contains a session token, so it should not be published.
Check the result by eye afterwards: the detection is a colour heuristic.
"""
import argparse
import shutil
from pathlib import Path

from PIL import Image

SHOTS = Path(__file__).resolve().parent.parent / "screenshots"
BACKUP = Path.home() / "screenshot_backup"
CROP_TOP = 72  # pixels: browser title bar + address bar


def has_lab_bar(image):
    strip = image.convert("RGB").crop((0, 0, image.width, 26))
    red, _green, blue = strip.resize((1, 1), Image.BOX).getpixel((0, 0))
    return 195 <= red <= 225 and blue - red >= 8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="crop the flagged screenshots")
    args = parser.parse_args()
    for path in sorted(SHOTS.glob("*")):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        with Image.open(path) as image:
            image.load()
            flagged = has_lab_bar(image)
            cropped = image.crop((0, CROP_TOP, image.width, image.height)) if flagged else None
        print(("CROP  " if flagged else "ok    ") + path.name)
        if flagged and args.apply:
            BACKUP.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, BACKUP / path.name)
            cropped.save(path)


if __name__ == "__main__":
    main()