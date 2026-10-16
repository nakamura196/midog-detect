# val の切片で、正解（緑=分裂像、黄=似た別物）と予測（赤=分裂像、青=似た別物）を重ねて並べる
import glob, os, random, sys
from PIL import Image, ImageDraw
from ultralytics import YOLO

root = os.path.expanduser("~/midog-detect/data/midog")
labs = sorted(glob.glob(f"{root}/labels/val/*.txt"))
by_spec = {}
for f in labs:
    if any(l.startswith("0 ") for l in open(f)):
        by_spec.setdefault(os.path.basename(f)[:3], []).append(f)
random.seed(1)
specs = sorted(by_spec)
pick = [random.choice(by_spec[s]) for s in specs[:: max(1, len(specs) // 7)]][:7]
empty = [f for f in labs if os.path.getsize(f) == 0]
pick.append(random.choice(empty) if empty else labs[0])

model = YOLO(sys.argv[1])
tiles = []
for lf in pick:
    name = os.path.basename(lf)[:-4]
    img = Image.open(f"{root}/images/val/{name}.jpg").convert("RGB")
    d = ImageDraw.Draw(img)
    W, H = img.size
    for l in open(lf):
        c, x, y, w, h = map(float, l.split())
        d.rectangle([(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H],
                    outline=(0, 200, 0) if c == 0 else (230, 200, 0), width=4)
    r = model.predict(f"{root}/images/val/{name}.jpg", imgsz=640, conf=0.25, verbose=False)[0]
    for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
        col = (230, 0, 0) if c == 0 else (0, 90, 255)
        d.rectangle([b[0] - 9, b[1] - 9, b[2] + 9, b[3] + 9], outline=col, width=4)
        d.text((b[0] - 9, b[3] + 11), f"{s:.2f}", fill=col, font_size=22)
    d.text((8, 8), name, fill=(0, 0, 0))
    tiles.append(img)
grid = Image.new("RGB", (640 * 4, 640 * 2), "white")
for i, t in enumerate(tiles):
    grid.paste(t, ((i % 4) * 640, (i // 4) * 640))
grid.save(os.path.expanduser("~/midog-detect/runs/pred_examples.jpg"), quality=88)
print("\n".join(os.path.basename(p) for p in pick))
