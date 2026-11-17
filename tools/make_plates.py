"""Print every post thumbnail as a wine-ink plate (duotone on paper), so no browser has to do it with CSS blend modes.
Run from the repo root after adding a post image:  python3 tools/make_plates.py
Writes assets/plates/<image path with / replaced by _>.jpg, 480 x 360, cropped like the list's 4:3 frame."""
import os, re, glob
from PIL import Image, ImageOps
PAPER = (0xED, 0xEF, 0xEC); WINE_DEEP = (0x5B, 0x14, 0x26)
def plate_name(img): return 'assets/plates/' + img.strip('/').replace('/', '_') + '.jpg'
images = set()
for f in glob.glob('_posts/**/*.md', recursive=True) + glob.glob('_posts/**/*.markdown', recursive=True):
    m = re.search(r'^image:\s*(\S+)', open(f, encoding='utf-8').read().split('---')[1] if open(f, encoding='utf-8').read().startswith('---') else '', re.M)
    if m: images.add(m.group(1).strip('"\''))
made = 0
for img in sorted(images):
    src = img.strip('/')
    if not os.path.exists(src): print('missing', src); continue
    im = Image.open(src).convert('RGB')
    im = ImageOps.fit(im, (480, 360), Image.LANCZOS, centering=(0.5, 0.5))
    g = ImageOps.grayscale(im); g = ImageOps.autocontrast(g, cutoff=1)
    lut = [[int(round(PAPER[c] / 255 * (WINE_DEEP[c] + (255 - WINE_DEEP[c]) * v / 255))) for v in range(256)] for c in range(3)]
    out = Image.merge('RGB', [g.point(lut[c]) for c in range(3)])
    out.save(plate_name(img), quality=86, optimize=True, progressive=True); made += 1
print(made, 'plates written to assets/plates/')
