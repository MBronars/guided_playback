"""Page GIF from gt_NNN.png / pred_NNN.png: two 320 px panels, a thin frame-counter strip, labels left to the HTML.
    python scripts/render_water3d/compose_page_gif.py <render dir> <context_len> <out.gif>"""
import glob, os, sys
from PIL import Image, ImageDraw, ImageFont
d, C, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
T = len(glob.glob(os.path.join(d, "gt_*.png"))); S, GAP, TOP = 320, 8, 26
try:
    import matplotlib; font = ImageFont.truetype(os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf/DejaVuSans.ttf"), 15)
except Exception:
    font = ImageFont.load_default()
frames = []
for t in range(T):
    L = Image.open(os.path.join(d, f"gt_{t:03d}.png")).convert("RGB").resize((S, S), Image.LANCZOS)
    R = Image.open(os.path.join(d, f"pred_{t:03d}.png")).convert("RGB").resize((S, S), Image.LANCZOS)
    im = Image.new("RGB", (2 * S + GAP, S + TOP), (250, 248, 244)); im.paste(L, (0, TOP)); im.paste(R, (S + GAP, TOP))
    tag = f"context frame {t + 1}/{C}" if t < C else f"predicted frame {t - C + 1}/{T - C}"
    ImageDraw.Draw(im).text((6, 4), tag, fill=(60, 60, 60), font=font)
    frames.append(im.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
frames[0].save(out, save_all=True, append_images=frames[1:], duration=[400] * C + [220] * (T - C), loop=0, optimize=True)
print(out, "%.2f MB" % (os.path.getsize(out) / 1e6), T, "frames", frames[0].size)
