"""Generate labeling.html: galeri patch + pelabelan manual + export CSV."""

import argparse
import base64
import csv
import json
import math
import random
from pathlib import Path

from config import CITIES, PATCH_SIZE
from preprocess import city_slug, output_tile_dir

SAR_FILES = [("T1", "SAR T1 · ~1.5×2.0 km"), ("T5", "SAR T5 · ~1.5×2.0 km")]
OPT_FILES = [
    ("NAIP_2020", "NAIP 2020"),
    ("NAIP_2022", "NAIP 2022"),
    ("S2_2020", "Sentinel-2 2020-11 · ~1.3 km"),
    ("S2_2023", "Sentinel-2 2023-09 · ~1.3 km"),
]


def _data_uri(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def build_items(city, shuffle_seed=123, embed=False):
    base = output_tile_dir(city)
    meta = json.loads((base / "prep" / "meta.json").read_text())
    with open(base / "samples" / "samples.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    ctx_path = base / "labeling" / "optical_context.json"
    ctx_data = json.loads(ctx_path.read_text()) if ctx_path.exists() else {}
    ctx = ctx_data.get("positions", {})
    naip_ground_m = float(ctx_data.get("target_km", 0.77)) * 1000.0

    h, w = int(meta["tile"]["h"]), int(meta["tile"]["w"])
    patch = PATCH_SIZE
    gt = meta["geotransform"]
    lat_mid = gt["north"] + 0.5 * h * gt["dy"]
    pix_lon_m = abs(gt["dx"]) * 111320.0 * math.cos(math.radians(lat_mid))
    pix_lat_m = abs(gt["dy"]) * 110574.0
    sar_w = patch * pix_lon_m
    sar_h = patch * pix_lat_m
    s2_ground_m = patch * 10.0
    # samakan lebar ground crosshair (arah horizontal) dengan SAR
    xh_scale_naip = sar_w / naip_ground_m if naip_ground_m > 0 else 1.0
    xh_scale_s2 = sar_w / s2_ground_m
    items = []
    for r in rows:
        row, col = int(r["row"]), int(r["col"])
        top = max(0, min(row - patch // 2, h - patch))
        left = max(0, min(col - patch // 2, w - patch))
        cx_sar, cy_sar = col - left, row - top
        imgs = []
        for key, label in SAR_FILES:
            p = base / "patches" / f"{r['id']}_{key}.png"
            if p.exists():
                src = _data_uri(p) if embed else "../patches/" + p.name
                imgs.append({"kind": "sar", "label": label, "src": src,
                             "cx_pct": 100.0 * cx_sar / patch,
                             "cy_pct": 100.0 * cy_sar / patch,
                             "xh_scale": 1.0})
        for key, label in OPT_FILES:
            p = base / "patches" / f"{r['id']}_{key}.png"
            if not p.exists():
                continue
            if key.startswith("NAIP"):
                cp = base / "patches" / f"{r['id']}_{key}_ctx.png"
                if cp.exists():
                    cpos = ctx.get(r["id"], {}).get(key, {"cx_pct": 50.0, "cy_pct": 50.0})
                    imgs.append({
                        "kind": "opt",
                        "label": f"{label} · ~{naip_ground_m:.0f} m",
                        "src": _data_uri(cp) if embed else "../patches/" + cp.name,
                        "cx_pct": cpos["cx_pct"], "cy_pct": cpos["cy_pct"],
                        "xh_scale": xh_scale_naip,
                    })
                    continue
            src = _data_uri(p) if embed else "../patches/" + p.name
            imgs.append({"kind": "opt", "label": label, "src": src,
                         "cx_pct": 50.0, "cy_pct": 50.0, "xh_scale": xh_scale_s2})
        items.append({
            "id": r["id"],
            "row": int(r["row"]), "col": int(r["col"]),
            "global_row": int(r["global_row"]), "global_col": int(r["global_col"]),
            "lat": float(r["lat"]), "lon": float(r["lon"]),
            "stratum": r["stratum"],
            "images": imgs,
        })

    random.Random(shuffle_seed).shuffle(items)
    dates = {k: v["date"] for k, v in meta["datasets"].items()}
    return items, dates, meta


TEMPLATE = r"""<!doctype html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Labeling __CITY__ - Change Detection</title>
<style>
  :root { --bg:#0f172a; --card:#111827; --fg:#e5e7eb; --muted:#9ca3af;
          --ok:#22c55e; --no:#ef4444; --warn:#f59e0b; --acc:#38bdf8; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         background:var(--bg); color:var(--fg); }
  header { position:sticky; top:0; background:#0b1220ee; backdrop-filter:blur(6px);
           padding:10px 16px; border-bottom:1px solid #1f2937; z-index:10; }
  .row { display:flex; align-items:center; gap:12px; flex-wrap:wrap; }
  h1 { font-size:16px; margin:0; }
  .muted { color:var(--muted); font-size:12px; }
  #progressbar { flex:1; height:8px; background:#1f2937; border-radius:6px; overflow:hidden; min-width:120px; }
  #progressfill { height:100%; width:0; background:var(--acc); transition:width .2s; }
  main { max-width:1500px; margin:16px auto; padding:0 20px; }
  .rubric { background:#111827; border:1px solid #1f2937; border-radius:10px;
            padding:10px 14px; font-size:13px; color:var(--muted); margin-bottom:12px; }
  .imgs { display:grid; grid-template-columns:repeat(2, minmax(0, 1fr));
          gap:16px; margin-bottom:14px; }
  @media (max-width: 900px) { .imgs { grid-template-columns:1fr; } }
  figure { margin:0; background:#000; border:1px solid #1f2937; border-radius:8px; overflow:hidden; }
  figure img { width:100%; height:auto; display:block; image-rendering:auto; }
  figure.sar img { image-rendering:pixelated; }
  figcaption { font-size:14px; color:var(--muted); padding:6px 8px; background:#0b1220; }
  .imgwrap { position:relative; line-height:0; }
  .xh { position:absolute; width:0; height:0;
        transform:scale(var(--s,1)); transform-origin:0 0;
        pointer-events:none; z-index:2; }
  .xh i { position:absolute; background:#ff00ff; box-shadow:0 0 0 1px rgba(0,0,0,.9); }
  .xh .l { width:40px; height:3px; right:12px; top:-1.5px; }
  .xh .r { width:40px; height:3px; left:12px; top:-1.5px; }
  .xh .t { width:3px; height:40px; bottom:12px; left:-1.5px; }
  .xh .b { width:3px; height:40px; top:12px; left:-1.5px; }
  .xh .c { width:34px; height:34px; border:2px solid #ff00ff; border-radius:50%;
           left:-17px; top:-17px; background:transparent;
           box-shadow:0 0 0 1px rgba(0,0,0,.7); }
  .hide-xh .xh { display:none; }
  .meta { font-size:13px; color:var(--muted); margin:8px 0; }
  .meta a { color:var(--acc); text-decoration:none; margin-right:12px; }
  .buttons { display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; }
  button { border:0; border-radius:8px; padding:10px 14px; font-size:14px; cursor:pointer; }
  button.ok { background:var(--ok); color:#04140a; }
  button.no { background:var(--no); color:#1a0505; }
  button.warn { background:var(--warn); color:#1a1203; }
  button.ghost { background:#1f2937; color:var(--fg); }
  input[type=text] { width:100%; padding:8px; border-radius:8px; border:1px solid #1f2937;
                     background:#0b1220; color:var(--fg); margin-top:6px; }
  .legend { font-size:12px; color:var(--muted); margin-top:8px; }
  #done { display:none; background:var(--card); border:1px solid #1f2937; border-radius:12px; padding:20px; }
  #done h2 { margin-top:0; }
  .stat { font-size:14px; margin:4px 0; }
</style>
</head>
<body>
<header>
  <div class="row">
    <h1>Labeling __CITY__</h1>
    <span class="muted" id="counter"></span>
    <div id="progressbar"><div id="progressfill"></div></div>
    <button class="ghost" id="xhbtn" onclick="toggleXH()">Crosshair (Space): ON</button>
    <button class="ghost" onclick="exportCSV()">Export CSV</button>
  </div>
</header>
<main>
  <div class="rubric">
    <b>Rubrik:</b> pilih <b>Berubah</b> bila ada perubahan tutupan/penggunaan lahan
    yang terlihat antara T1 (__DATES_T1__) dan T5 (__DATES_T5__): bangunan baru,
    jalan, tebangan, vegetasi hilang, alur/bibir sungai. Bukan perbedaan musim,
    kelembapan tanah, atau speckle. T1/T5 = SAR RGB palsu (HH/HV/VV); optik hanya
    pembantu. Crosshair magenta = titik sampel (tengahnya sengaja dikosongkan).
    NAIP ditampilkan pada cakupan ~770 m (10x lebih luas dari detail 77 m);
    ukuran crosshair NAIP diskalakan agar cakupan tanah yang ditunjuk sama dengan SAR.
  </div>

  <section id="card">
    <div class="meta" id="samplemeta"></div>
    <div class="imgs" id="imgs"></div>
    <input type="text" id="note" placeholder="Catatan (opsional): mis. 'bangunan baru', 'tebangan', 'ragu karena awan'...">
    <div class="buttons">
      <button class="ok" onclick="setLabel('1')">1 - Berubah</button>
      <button class="no" onclick="setLabel('0')">0 - Tidak</button>
      <button class="warn" onclick="setLabel('u')">u - Ragu</button>
      <button class="ghost" onclick="setLabel('')">s - Lewati</button>
      <button class="ghost" onclick="back()">← Kembali</button>
    </div>
    <div class="legend">Shortcut: 1 / 0 / u / s / ← (backspace). Space: crosshair on/off.
      Label disimpan otomatis di browser.</div>
  </section>

  <section id="done">
    <h2>Selesai</h2>
    <div id="summary"></div>
    <p class="muted">Klik Export CSV, lalu gabungkan dengan kota lain menjadi
      <code>labels/L1_labels.csv</code>.</p>
    <div class="buttons">
      <button class="ok" onclick="exportCSV()">Export CSV</button>
      <button class="ghost" onclick="back()">← Kembali ke sampel terakhir</button>
    </div>
  </section>
</main>
<script>
const CITY = __CITY__;
const SLUG = __SLUG__;
const SAMPLES = __SAMPLES__;
const KEY = "labels_L1_" + SLUG;
let labels = {};
try { labels = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { labels = {}; }
let idx = 0;

const el = (id) => document.getElementById(id);
let showXH = true;
function toggleXH() {
  showXH = !showXH;
  document.body.classList.toggle("hide-xh", !showXH);
  el("xhbtn").textContent = "Crosshair (Space): " + (showXH ? "ON" : "OFF");
}
function save() { localStorage.setItem(KEY, JSON.stringify(labels)); }
function labeledCount() { return SAMPLES.filter(s => labels[s.id] && labels[s.id].label !== undefined && labels[s.id].label !== "").length; }

function render() {
  const done = idx >= SAMPLES.length;
  el("card").style.display = done ? "none" : "block";
  el("done").style.display = done ? "block" : "none";
  el("counter").textContent = `${Math.min(idx + 1, SAMPLES.length)} / ${SAMPLES.length} (${labeledCount()} berlabel)`;
  el("progressfill").style.width = `${100 * labeledCount() / SAMPLES.length}%`;
  if (done) { renderSummary(); return; }

  const s = SAMPLES[idx];
  const prev = labels[s.id] || {};
  el("note").value = prev.note || "";
  const ge = `https://earth.google.com/web/@${s.lat},${s.lon},0a,1200d,35y,0h,0t,0r`;
  const eo = `https://apps.sentinel-hub.com/eo-browser/?lat=${s.lat}&lng=${s.lon}&zoom=15`;
  el("samplemeta").innerHTML =
    `<b>${s.id}</b> · tile row=${s.row}, col=${s.col} · ` +
    `lat=${s.lat.toFixed(6)}, lon=${s.lon.toFixed(6)} · strata=${s.stratum} · ` +
    `<a href="${ge}" target="_blank">Google Earth</a>` +
    `<a href="${eo}" target="_blank">EO Browser</a>`;
  el("imgs").innerHTML = s.images.map(im => {
    const x = (im.cx_pct ?? 50).toFixed(3);
    const y = (im.cy_pct ?? 50).toFixed(3);
    const sc = (im.xh_scale ?? 1).toFixed(3);
    return `<figure class="${im.kind}"><div class="imgwrap">` +
      `<img loading="lazy" src="${im.src}" alt="${im.label}">` +
      `<div class="xh" style="left:${x}%;top:${y}%;--s:${sc}">` +
      `<i class="l"></i><i class="r"></i><i class="t"></i><i class="b"></i><i class="c"></i>` +
      `</div></div><figcaption>${im.label}</figcaption></figure>`;
  }).join("");
}

function setLabel(v) {
  if (idx >= SAMPLES.length) return;
  const s = SAMPLES[idx];
  labels[s.id] = { label: v, note: el("note").value, ts: new Date().toISOString() };
  save();
  idx += 1;
  render();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function back() {
  idx = Math.max(0, idx - 1);
  render();
}

function renderSummary() {
  let n1 = 0, n0 = 0, nu = 0, nempty = 0;
  for (const s of SAMPLES) {
    const v = labels[s.id] ? labels[s.id].label : undefined;
    if (v === "1") n1++; else if (v === "0") n0++; else if (v === "u") nu++; else nempty++;
  }
  el("summary").innerHTML =
    `<div class="stat">Berubah: <b>${n1}</b></div>` +
    `<div class="stat">Tidak: <b>${n0}</b></div>` +
    `<div class="stat">Ragu: <b>${nu}</b></div>` +
    `<div class="stat">Kosong/lewati: <b>${nempty}</b></div>`;
}

function csvEscape(x) {
  const s = String(x == null ? "" : x);
  return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
}

function exportCSV() {
  const header = ["id","city","row","col","global_row","global_col","lat","lon","label","note","timestamp"];
  const lines = [header.join(",")];
  for (const s of SAMPLES) {
    const l = labels[s.id] || {};
    const vals = [s.id, CITY, s.row, s.col, s.global_row, s.global_col,
                  s.lat, s.lon, l.label ?? "", l.note ?? "", l.ts ?? ""];
    lines.push(vals.map(csvEscape).join(","));
  }
  const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `labels_${SLUG}.csv`;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

document.addEventListener("keydown", (e) => {
  if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA")) return;
  if (e.code === "Space") { e.preventDefault(); toggleXH(); return; }
  if (e.key === "1") { e.preventDefault(); setLabel("1"); }
  else if (e.key === "0") { e.preventDefault(); setLabel("0"); }
  else if (e.key.toLowerCase() === "u") { e.preventDefault(); setLabel("u"); }
  else if (e.key.toLowerCase() === "s") { e.preventDefault(); setLabel(""); }
  else if (e.key === "Backspace" || e.key === "ArrowLeft") { e.preventDefault(); back(); }
});

render();
</script>
</body>
</html>
"""


def build_html(city, items, dates, out_path, shuffle_seed=123):
    samp = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    html = (TEMPLATE
            .replace("__CITY__", json.dumps(city))
            .replace("__SLUG__", json.dumps(city_slug(city)))
            .replace("__SAMPLES__", samp)
            .replace("__DATES_T1__", dates.get("T1", "2020"))
            .replace("__DATES_T5__", dates.get("T5", "2023")))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def label_city(city, shuffle_seed=123, embed=False):
    base = output_tile_dir(city)
    items, dates, _ = build_items(city, shuffle_seed, embed=embed)
    out = base / "labeling" / "labeling.html"
    return build_html(city, items, dates, out, shuffle_seed), len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--embed", action="store_true", help="embed gambar base64 (HTML besar)")
    a = ap.parse_args()
    for city in (a.cities or list(CITIES)):
        p, n = label_city(city, a.seed, embed=a.embed)
        print(f"{city}: {n} sampel -> {p} ({p.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
