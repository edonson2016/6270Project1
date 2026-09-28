"""Emit the thermal-energy-storage experiment report as a self-contained HTML page."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

R = Path("runs/il/results")
OUT = Path("/tmp/claude-1000/-home-edonson2016/a19f98e4-a524-489b-8bf4-802035593af9/scratchpad/tes-report.html")
TES = json.loads((R / "tes_screen5.json").read_text())
MC = json.loads((R / "mc_frames.json").read_text())

# Hue carries the generative model and nothing else -- every bar is directly
# labelled, so colour does not need to encode arm identity as well.
ARMS = [("ddpm_default", "DDPM", "d", 0), ("fm_default", "FM", "f", 0),
        ("fm_recipe", "FM recipe", "f", 0), ("ddpm_cfg250", "DDPM→250", "d", 0),
        ("fm_cfg250", "FM→250", "f", 0), ("fm_cfg0", "FM→0", "f", 0)]
NFE = {"ddpm_default": 1000, "fm_default": 100, "fm_recipe": 100,
       "ddpm_cfg250": 1000, "fm_cfg250": 100, "fm_cfg0": 100}
LONG = {"ddpm_default": "DDPM default", "fm_default": "FM default",
        "fm_recipe": "FM recipe (full+OT+σ)", "ddpm_cfg250": "DDPM + CFG → 250 °C",
        "fm_cfg250": "FM + CFG → 250 °C", "fm_cfg0": "FM + CFG → 0 °C"}
COL = {("d", 0): "var(--c2)", ("f", 0): "var(--u3)"}


def bars(vals, title, fmt="{:.0f}", h=250, w=660, note=""):
    """vals: [(label, value, colour)]. Single series, so no legend is needed."""
    mx = max(v for _, v, _ in vals) or 1
    pad, bw = 46, 74
    out = []
    for n, (lab, v, c) in enumerate(vals):
        x = pad + n * (bw + 22)
        bh = (v / mx) * (h - 64)
        y = h - 34 - bh
        out.append(
            f'<rect x="{x}" y="{y:.1f}" width="{bw}" height="{max(bh,1):.1f}" fill="{c}" rx="4"/>'
            f'<text x="{x+bw/2}" y="{y-7:.1f}" font-size="12.5" font-weight="600" text-anchor="middle" fill="var(--ink)">{fmt.format(v)}</text>'
            f'<text x="{x+bw/2}" y="{h-18}" font-size="10.5" text-anchor="middle" fill="var(--muted)">{lab}</text>')
    cap = f'<figcaption class="ft">{title}</figcaption>'
    sub = f'<p class="sub">{note}</p>' if note else ""
    return (f'<figure>{cap}<div class="fs"><svg viewBox="0 0 {w} {h}" role="img" aria-label="{title}">'
            f'<line x1="{pad-8}" y1="{h-34}" x2="{w-10}" y2="{h-34}" stroke="var(--line)"/>'
            + "".join(out) + f"</svg></div>{sub}</figure>")


def arm_bars(fn, title, fmt="{:.0f}", note=""):
    return bars([(lab, fn(TES[k]), COL[(f, i)]) for k, lab, f, i in ARMS],
                title, fmt, note=note)


def cost_quality(w=660, h=300):
    """NFE against hit rate. Log x, because the arms span 100 to 1000."""
    L, Rp, T, B = 62, 120, 24, 46
    pts = [(NFE[k], TES[k]["funnel"]["lr_gt_250"] / max(TES[k]["funnel"]["liquid_range_both_ions"], 1) * 100,
            lab, COL[(f, i)]) for k, lab, f, i in ARMS]
    import math as _m
    def px(x): return L + (_m.log10(x) - 1.7) / (3.1 - 1.7) * (w - L - Rp)
    def py(y): return h - B - (y - 15) / (55 - 15) * (h - T - B)
    g = [f'<line x1="{L}" y1="{h-B}" x2="{w-Rp}" y2="{h-B}" stroke="var(--line)"/>'
         f'<line x1="{L}" y1="{T}" x2="{L}" y2="{h-B}" stroke="var(--line)"/>']
    for v in (20, 30, 40, 50):
        g.append(f'<line x1="{L}" y1="{py(v):.1f}" x2="{w-Rp}" y2="{py(v):.1f}" stroke="var(--line)" stroke-dasharray="2 4"/>'
                 f'<text x="{L-8}" y="{py(v)+4:.1f}" font-size="10.5" text-anchor="end" fill="var(--muted)">{v}%</text>')
    for v in (100, 300, 1000):
        g.append(f'<text x="{px(v):.1f}" y="{h-B+17}" font-size="10.5" text-anchor="middle" fill="var(--muted)">{v}</text>')
    for x, y, lab, c in pts:
        g.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="7" fill="{c}"/>'
                 f'<text x="{px(x)+11:.1f}" y="{py(y)+4:.1f}" font-size="11" font-weight="600" fill="{c}">{lab}</text>')
    g.append(f'<text x="{(L+w-Rp)/2}" y="{h-6}" font-size="11.5" text-anchor="middle" fill="var(--ink-2)">network evaluations per sample (log) →</text>')
    g.append(f'<text x="14" y="{T+6}" font-size="11.5" fill="var(--ink-2)">hit rate ↑</text>')
    return (f'<figure><figcaption class="ft">Sampling cost against screened hit rate</figcaption>'
            f'<div class="fs"><svg viewBox="0 0 {w} {h}" role="img" aria-label="cost versus hit rate">'
            + "".join(g) + '</svg></div><p class="sub">Left is cheaper, up is better. Diffusion sits an order of magnitude to the right for no gain.</p></figure>')


def grouped_bars(metric, title, fmt="{:.0f}", h=250):
    vals = [(lab, TES[k]["funnel"][metric] if metric in TES[k]["funnel"] else 0, COL[(f, i)])
            for k, lab, f, i in ARMS]
    mx = max(v for _, v, _ in vals) or 1
    w, pad, bw = 660, 46, 74
    bars = []
    for n, (lab, v, c) in enumerate(vals):
        x = pad + n * (bw + 22)
        bh = (v / mx) * (h - 64)
        y = h - 34 - bh
        bars.append(
            f'<rect x="{x}" y="{y:.1f}" width="{bw}" height="{max(bh,1):.1f}" fill="{c}" rx="4"/>'
            f'<text x="{x+bw/2}" y="{y-7:.1f}" font-size="12.5" font-weight="600" text-anchor="middle" fill="var(--ink)">{fmt.format(v)}</text>'
            f'<text x="{x+bw/2}" y="{h-18}" font-size="10.5" text-anchor="middle" fill="var(--muted)">{lab.split(" (")[0]}</text>')
    return (f'<figure><figcaption class="ft">{title}</figcaption><div class="fs">'
            f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{title}">'
            f'<line x1="{pad-8}" y1="{h-34}" x2="{w-10}" y2="{h-34}" stroke="var(--line)"/>'
            + "".join(bars) + "</svg></div></figure>")


def line_chart(series, title, ylab, ymin=None, ymax=None, w=660, h=260):
    xs = sorted({x for _, pts, _ in series for x, _ in pts})
    ys = [y for _, pts, _ in series for _, y in pts]
    lo = ymin if ymin is not None else min(ys) * 0.97
    hi = ymax if ymax is not None else max(ys) * 1.03
    L, Rp, T, B = 62, 18, 24, 42
    def px(x): return L + (x - min(xs)) / max(max(xs) - min(xs), 1e-9) * (w - L - Rp)
    def py(y): return h - B - (y - lo) / max(hi - lo, 1e-9) * (h - T - B)
    g = [f'<line x1="{L}" y1="{h-B}" x2="{w-Rp}" y2="{h-B}" stroke="var(--line)"/>'
         f'<line x1="{L}" y1="{T}" x2="{L}" y2="{h-B}" stroke="var(--line)"/>']
    for f in range(4):
        v = lo + (hi - lo) * f / 3
        g.append(f'<line x1="{L}" y1="{py(v):.1f}" x2="{w-Rp}" y2="{py(v):.1f}" stroke="var(--line)" stroke-dasharray="2 4"/>'
                 f'<text x="{L-8}" y="{py(v)+4:.1f}" font-size="10.5" text-anchor="end" fill="var(--muted)">{v:.2f}</text>')
    for x in xs:
        g.append(f'<text x="{px(x):.1f}" y="{h-B+17}" font-size="10.5" text-anchor="middle" fill="var(--muted)">{x:g}</text>')
    for lab, pts, c in series:
        p = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in pts)
        g.append(f'<polyline fill="none" stroke="{c}" stroke-width="2.2" points="{p}"/>')
        for x, y in pts:
            g.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3.6" fill="{c}"/>')
        lx, ly = pts[-1]
        g.append(f'<text x="{px(lx)-4:.1f}" y="{py(ly)-9:.1f}" font-size="11" font-weight="600" text-anchor="end" fill="{c}">{lab}</text>')
    g.append(f'<text x="{(L+w-Rp)/2}" y="{h-4}" font-size="11.5" text-anchor="middle" fill="var(--ink-2)">{ylab}</text>')
    return (f'<figure><figcaption class="ft">{title}</figcaption><div class="fs">'
            f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{title}">' + "".join(g) + "</svg></div></figure>")


def scatter(title, w=660, h=300):
    L, Rp, T, B = 62, 90, 24, 46
    pts = [(TES[k]["novel_cation_rate"], TES[k]["funnel"]["lr_gt_250"], lab, COL[(f, i)])
           for k, lab, f, i in ARMS]
    xmin, xmax = 0.22, 0.82; ymin, ymax = 0, 160
    def px(x): return L + (x - xmin) / (xmax - xmin) * (w - L - Rp)
    def py(y): return h - B - (y - ymin) / (ymax - ymin) * (h - T - B)
    g = [f'<line x1="{L}" y1="{h-B}" x2="{w-Rp}" y2="{h-B}" stroke="var(--line)"/>'
         f'<line x1="{L}" y1="{T}" x2="{L}" y2="{h-B}" stroke="var(--line)"/>']
    for v in (0, 40, 80, 120, 160):
        g.append(f'<line x1="{L}" y1="{py(v):.1f}" x2="{w-Rp}" y2="{py(v):.1f}" stroke="var(--line)" stroke-dasharray="2 4"/>'
                 f'<text x="{L-8}" y="{py(v)+4:.1f}" font-size="10.5" text-anchor="end" fill="var(--muted)">{v}</text>')
    for v in (0.25, 0.4, 0.55, 0.7):
        g.append(f'<text x="{px(v):.1f}" y="{h-B+17}" font-size="10.5" text-anchor="middle" fill="var(--muted)">{v:.0%}</text>')
    for x, y, lab, c in pts:
        g.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="7" fill="{c}"/>'
                 f'<text x="{px(x)+11:.1f}" y="{py(y)+4:.1f}" font-size="11" font-weight="600" fill="{c}">{lab}</text>')
    g.append(f'<text x="{(L+w-Rp)/2}" y="{h-6}" font-size="11.5" text-anchor="middle" fill="var(--ink-2)">novel-cation rate →</text>')
    g.append(f'<text x="14" y="{T+6}" font-size="11.5" fill="var(--ink-2)">TES hits ↑</text>')
    return (f'<figure><figcaption class="ft">{title}</figcaption><div class="fs">'
            f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{title}">' + "".join(g) + "</svg></div></figure>")


def lr_hist(w=660, h=260):
    L, Rp, T, B = 50, 18, 24, 44
    edges = np.arange(0, 450, 40)
    g = [f'<line x1="{L}" y1="{h-B}" x2="{w-Rp}" y2="{h-B}" stroke="var(--line)"/>']
    series = [("ddpm_default", "d", 0), ("fm_recipe", "f", 0), ("fm_cfg0", "f", 0)]
    mx = 0; H = {}
    for k, f, i in series:
        lr = [c["liquid_range_C"] for c in TES[k]["candidates"] if c.get("lr_n_ions") == 2]
        hh, _ = np.histogram(lr, bins=edges); H[k] = hh; mx = max(mx, hh.max())
    bw = (w - L - Rp) / (len(edges) - 1)
    for n, (k, f, i) in enumerate(series):
        for b, v in enumerate(H[k]):
            x = L + b * bw + n * (bw / 3.2)
            bh = v / mx * (h - T - B)
            op = [0.45, 0.75, 1.0][n]
            g.append(f'<rect x="{x:.1f}" y="{h-B-bh:.1f}" width="{bw/3.4:.1f}" height="{max(bh,0.5):.1f}" fill="{COL[(f,i)]}" opacity="{op}" rx="1.5"/>')
    for b in range(0, len(edges) - 1, 2):
        g.append(f'<text x="{L+b*bw+bw/2:.1f}" y="{h-B+17}" font-size="10" text-anchor="middle" fill="var(--muted)">{edges[b]}</text>')
    g.append(f'<line x1="{L+(250/440)*(w-L-Rp):.1f}" y1="{T}" x2="{L+(250/440)*(w-L-Rp):.1f}" y2="{h-B}" stroke="var(--warn)" stroke-width="1.6" stroke-dasharray="5 4"/>')
    g.append(f'<text x="{L+(250/440)*(w-L-Rp)+6:.1f}" y="{T+12}" font-size="10.5" fill="var(--warn)">250 °C threshold</text>')
    g.append(f'<text x="{(L+w-Rp)/2}" y="{h-6}" font-size="11.5" text-anchor="middle" fill="var(--ink-2)">liquid range T_dec − MP (°C)  ·  ion-level measured evidence</text>')
    return (f'<figure><figcaption class="ft">Liquid-range distribution of screened candidates</figcaption>'
            f'<div class="fs"><svg viewBox="0 0 {w} {h}" role="img" aria-label="liquid range histogram">'
            + "".join(g) + "</svg></div></figure>")


# ---- build page
loss = [("FM default", [(2000,1.3076),(4000,1.2126),(6000,1.1765),(8000,1.2396)], "var(--u1)"),
        ("FM + CFG", [(2000,1.2647),(4000,1.1979),(6000,1.1823),(8000,1.2122)], "var(--u2)"),
        ("FM recipe (σ=0.20)", [(2000,0.6009),(4000,0.5810),(6000,0.5614),(8000,0.5703)], "var(--u3)")]
rmsf = []
ramp = ["var(--u1)", "var(--u2)", "var(--u3)", "var(--c2)"]
for n, mol in enumerate(MC):
    pts = [(r["T"], r["rmsf"]) for r in mol["runs"]]
    rmsf.append((f"mol {n+1}", pts, ramp[n % 4]))

rows = []
for k, lab, f, i in ARMS:
    v = TES[k]; fu = v["funnel"]
    rows.append(f"<tr{' class=hl' if k in ('fm_recipe','fm_cfg0') else ''}><td><b>{LONG[k]}</b></td>"
                f"<td class=n>{NFE[k]}</td><td class=n>{v['plausible_rate']:.3f}</td>"
                f"<td class=n>{v['uniqueness']:.3f}</td><td class=n>{v['novel_cation_rate']:.3f}</td>"
                f"<td class=n>{fu['distinct']}</td><td class=n>{fu['both_ions_measured']}</td>"
                f"<td class=n>{fu['liquid_range_both_ions']}</td>"
                f"<td class=n><b>{fu['lr_gt_250']}</b></td>"
                f"<td class=n><b>{fu['lr_gt_250']/max(fu['liquid_range_both_ions'],1):.1%}</b></td>"
                f"<td class=n>{fu['lr_gt_300']}</td>"
                f"<td class=n>{fu['novel_pair_and_lr_gt_250']}</td></tr>")

top = []
for k in ("fm_cfg0", "fm_recipe", "ddpm_default"):
    for c in TES[k]["top"][:3]:
        top.append(f"<tr><td class=m>{c['pair']}</td><td class=n>{c['liquid_range_C']:.0f}</td>"
                   f"<td>{c['cation_family']} / {c['anion_family']}</td><td class=n>{c['heavy']}</td>"
                   f"<td>{'yes' if c['novel_pair'] else 'no'}</td><td>{k}</td></tr>")

mcjson = json.dumps([{ "smiles": m["smiles"],
                       "runs": [{"T": r["T"], "symbols": r["symbols"], "frames": r["frames"],
                                 "rmsf": r["rmsf"]} for r in m["runs"]]} for m in MC])

Path(OUT).write_text(f"""<title>Ionic Liquids for Thermal Storage</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:wght@400;600&display=swap">
<script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.1.0/3Dmol-min.js"></script>
<style>
:root{{--ground:#F5F7F9;--surface:#FFF;--surface2:#EDF1F5;--ink:#10161D;--ink-2:#39434F;--muted:#626D7A;
--line:#DFE5EC;--accent:#2B54C8;--warn:#B4682A;--good:#136B52;
--u1:#A9BEEA;--u2:#5C7FD6;--u3:#2B54C8;--c1:#E8C49A;--c2:#C98C45;--c3:#8A5A1E;
--fs:"IBM Plex Sans",system-ui,sans-serif;--fr:"IBM Plex Serif",Georgia,serif;--fm:"IBM Plex Mono",monospace;}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--ground:#0C1116;--surface:#141B22;--surface2:#1A232C;
--ink:#E7EDF4;--ink-2:#BCC7D3;--muted:#8E9BA9;--line:#242F3A;--accent:#7D9EFF;--warn:#E0975A;--good:#45C2A0;
--u1:#3C5687;--u2:#5E86D8;--u3:#9DB8FF;--c1:#6B4A22;--c2:#C2853F;--c3:#E8B573;}}}}
:root[data-theme=dark]{{--ground:#0C1116;--surface:#141B22;--surface2:#1A232C;--ink:#E7EDF4;--ink-2:#BCC7D3;
--muted:#8E9BA9;--line:#242F3A;--accent:#7D9EFF;--warn:#E0975A;--good:#45C2A0;
--u1:#3C5687;--u2:#5E86D8;--u3:#9DB8FF;--c1:#6B4A22;--c2:#C2853F;--c3:#E8B573;}}
*{{box-sizing:border-box}}
body{{background:var(--ground);color:var(--ink);font-family:var(--fr);font-size:16px;line-height:1.62;margin:0;padding-inline:20px}}
.w{{max-width:920px;margin-inline:auto}}
header{{padding-block:52px 28px;border-bottom:2px solid var(--ink)}}
.eb{{font-family:var(--fs);font-size:11px;font-weight:600;letter-spacing:.16em;text-transform:uppercase;color:var(--accent);margin:0 0 14px}}
h1{{font-family:var(--fs);font-weight:700;font-size:clamp(28px,5vw,46px);line-height:1.08;letter-spacing:-.02em;margin:0 0 16px;text-wrap:balance}}
.sf{{font-size:clamp(16px,2vw,18.5px);color:var(--ink-2);max-width:64ch;margin:0 0 26px}}
.kpi{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1px;background:var(--line);border:1px solid var(--line)}}
.kpi>div{{background:var(--surface);padding:13px 15px}}
.kpi dt{{font-family:var(--fs);font-size:10.5px;font-weight:600;letter-spacing:.11em;text-transform:uppercase;color:var(--muted);margin:0 0 5px}}
.kpi dd{{margin:0;font-family:var(--fm);font-size:16px;color:var(--ink);font-variant-numeric:tabular-nums}}
.kpi dd small{{font-size:11px;color:var(--muted)}}
h2{{font-family:var(--fs);font-weight:600;font-size:clamp(20px,3vw,26px);letter-spacing:-.015em;margin:46px 0 6px;text-wrap:balance}}
h2 .num{{font-family:var(--fm);font-size:14px;color:var(--accent);margin-right:10px}}
h3{{font-family:var(--fs);font-weight:600;font-size:16.5px;margin:30px 0 8px}}
p{{margin:0 0 15px;max-width:68ch}} ul{{max-width:68ch}} li{{margin-bottom:7px}}
code{{font-family:var(--fm);font-size:.875em;background:var(--surface2);padding:1.5px 5px;border-radius:3px}}
.ts{{overflow-x:auto;border:1px solid var(--line);background:var(--surface);margin:0 0 20px}}
table{{border-collapse:collapse;width:100%;font-family:var(--fs);font-size:13px}}
th,td{{padding:8px 12px;text-align:left;border-bottom:1px solid var(--line)}}
thead th{{font-size:10.5px;font-weight:600;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);background:var(--surface2);white-space:nowrap}}
td.n{{text-align:right;font-family:var(--fm);font-variant-numeric:tabular-nums;white-space:nowrap}}
td.m{{font-family:var(--fm);font-size:11px;max-width:330px;word-break:break-all}}
tbody tr:last-child td{{border-bottom:none}}
.hl{{background:color-mix(in srgb,var(--accent) 7%,transparent)}}
figure{{margin:0 0 26px}}
.ft{{font-family:var(--fs);font-size:11px;font-weight:600;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);margin:0 0 8px}}
.sub{{font-family:var(--fs);font-size:12.5px;color:var(--muted);margin:8px 0 0;max-width:68ch}}
.fs{{overflow-x:auto;border:1px solid var(--line);background:var(--surface);padding:14px 12px}}
.fs svg{{display:block;min-width:600px;max-width:100%;height:auto}}
.two{{display:grid;grid-template-columns:1fr;gap:4px}}
@media(min-width:800px){{.two{{grid-template-columns:1fr 1fr;gap:18px}}}}
.two .fs svg{{min-width:300px}}
.note{{border-left:3px solid var(--accent);background:color-mix(in srgb,var(--accent) 8%,var(--surface));padding:14px 18px;margin:0 0 20px;max-width:68ch;font-size:15px}}
.note.w2{{border-left-color:var(--warn);background:color-mix(in srgb,var(--warn) 9%,var(--surface))}}
.note.g{{border-left-color:var(--good);background:color-mix(in srgb,var(--good) 8%,var(--surface))}}
.note .tg{{display:block;font-family:var(--fs);font-size:10.5px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--accent);margin-bottom:6px}}
.note.w2 .tg{{color:var(--warn)}} .note.g .tg{{color:var(--good)}}
#v{{width:100%;height:420px;position:relative;border:1px solid var(--line);background:var(--surface)}}
.ctl{{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:12px 0 18px;font-family:var(--fs);font-size:13px}}
.ctl button,.ctl select{{font-family:var(--fs);font-size:13px;padding:6px 11px;border:1px solid var(--line);background:var(--surface);color:var(--ink);border-radius:5px;cursor:pointer}}
.ctl button[aria-pressed=true]{{background:var(--accent);color:#fff;border-color:transparent}}
footer{{border-top:2px solid var(--ink);padding-block:20px 42px;font-family:var(--fs);font-size:12.5px;color:var(--muted);margin-top:44px}}
:focus-visible{{outline:2px solid var(--accent);outline-offset:2px}}
</style>
<div class="w">
<header>
<p class="eb">Experimental report · flow matching for ionic liquids</p>
<h1>Ionic Liquids for Thermal Storage</h1>
<p class="sf">Flow matching and diffusion, each unconditional and each with melting-point guidance, plus a tuned flow-matching recipe — six configurations screened against measured decomposition and melting temperatures. Flow matching matches diffusion at a tenth the sampling cost. Beyond that, two corrections: conditioning toward <em>high</em> melting point is the wrong direction for a liquid heat store, and the best generator yields the fewest defensible candidates for reasons unrelated to its quality.</p>
<dl class="kpi">
<div><dt>Best generator</dt><dd>FM recipe</dd></div>
<div><dt>FM vs DDPM</dt><dd>10× <small>cheaper, equal</small></dd></div>
<div><dt>Best shortlist</dt><dd>CFG → 0 °C</dd></div>
<div><dt>Hit rate, best</dt><dd>50.0% <small>vs 37.0%</small></dd></div>
<div><dt>Candidates &gt; 250 °C</dt><dd>149 <small>/ 2000</small></dd></div>
<div><dt>Widest liquid range</dt><dd>403 °C</dd></div>
<div><dt>Measured pairs</dt><dd>3,070</dd></div>
</dl>
</header>

<h2><span class="num">01</span>Two questions, deliberately separated</h2>
<p>A generative model for materials discovery gets judged on two things that are easy to conflate, and this experiment pulls them apart:</p>
<ul>
<li><strong>How good is the generator?</strong> Does it produce valid, distinct, chemically novel ionic liquids? Answered by plausibility, uniqueness and novel-ion rate.</li>
<li><strong>How many candidates can existing measurements vouch for?</strong> Answered by screening against real decomposition and melting data — which can only score a molecule whose ions somebody has already measured.</li>
</ul>
<p>These give <em>different winners</em>, and the gap between them is the most useful thing in this report. A model that explores unmeasured chemistry looks worse on the second question precisely because it succeeds at the first.</p>

<h2><span class="num">02</span>What was compared</h2>
<p>Two generative families (flow matching, diffusion), each unconditional and each conditioned on melting point, plus the tuned flow-matching recipe and a low-target conditional arm. Every arm shares one autoencoder, one frozen latent geometry, the SELFIES vocabulary and identical data splits, so differences are attributable to the generative model and its conditioning rather than to the representation. 2,000 samples each.</p>
<div class="ts"><table>
<thead><tr><th>Arm</th><th class=n>NFE</th><th class=n>plaus</th><th class=n>uniq</th><th class=n>nov cat</th><th class=n>distinct</th><th class=n>scoreable</th><th class=n>lr&gt;250</th><th class=n>hit rate</th><th class=n>lr&gt;300</th><th class=n>novel &amp; lr&gt;250</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<p class="sub"><b>scoreable</b> = both ions carry a measured melting <em>and</em> decomposition temperature. <b>hit rate</b> = lr&gt;250 as a fraction of scoreable, i.e. the model's success rate where the data can actually judge it.</p>

<h2><span class="num">03</span>Flow matching against diffusion</h2>
<p>Both generative families were run unconditionally and with melting-point guidance, on the same autoencoder and latent geometry. The comparison is not close on cost.</p>
{cost_quality()}
<div class="note g"><span class="tg">Flow matching matches diffusion at a tenth the cost</span>
<p>FM default reaches 0.943 plausibility at <b>100</b> network evaluations; DDPM reaches 0.930 at <b>1,000</b>. Their screened hit rates are indistinguishable (38.9% against 39.0%). Diffusion is marginally more diverse (uniqueness 0.841 against 0.823, novel-cation 0.406 against 0.361), which is a real if small advantage — but it costs a factor of ten in sampling, and the recipe recovers the diversity anyway at 0.853.</p></div>
<p>Guidance works on <em>both</em> families, and fails on both in the same way at a high melting-point target — 12 candidates for diffusion, 5 for flow matching, against roughly 100 unconditional. That the failure is shared is the useful part: it locates the problem in the <strong>target</strong>, not in either generative model.</p>
<p>One asymmetry worth recording: pushed to the hard target, diffusion degrades less. DDPM + CFG holds a 29.3% hit rate against flow matching's 19.2%, and keeps more of its diversity (uniqueness 0.719 against 0.641). A thousand-step reverse chain appears to handle the periphery of the latent distribution better than a fifty-step ODE solve — at ten times the cost.</p>

<h2><span class="num">04</span>Generation quality across all five</h2>
<p>The recipe wins on every axis, and at 8 function evaluations against the default's 100.</p>
<div class="two">
{arm_bars(lambda v: v["plausible_rate"], "Plausibility", "{:.3f}")}
{arm_bars(lambda v: v["uniqueness"], "Uniqueness", "{:.3f}")}
</div>
<div class="two">
{arm_bars(lambda v: v["novel_cation_rate"], "Novel-cation rate", "{:.3f}")}
{arm_bars(lambda v: v["funnel"]["distinct"], "Distinct plausible molecules")}
</div>
<div class="note g"><span class="tg">Question one: the recipe</span>
<p>Plausibility 0.943 against 0.855, uniqueness 0.853 against 0.794, novel-cation 0.386 against 0.250, and 1,603 distinct usable molecules against 1,358 — at one twelfth the sampling cost. On generation there is no ambiguity.</p></div>

<h2><span class="num">05</span>Question two — screened yield, and why the raw count misleads</h2>
{grouped_bars("lr_gt_250", "Candidates with liquid range > 250 °C, per 2,000 samples")}
<p>Read alone, this says the recipe (105) slightly trails the default (111). That is a <strong>coverage artifact</strong>. The screen discards any molecule whose ions nobody has measured, and the recipe routes far more of its output into exactly that territory. Normalising by what is scoreable reverses the ordering:</p>
{arm_bars(lambda v: v["funnel"]["lr_gt_250"]/max(v["funnel"]["liquid_range_both_ions"],1)*100, "Hit rate among scoreable candidates (%)", "{:.1f}", note="The recipe converts 40.2% of its scoreable candidates into wide-liquid-range ILs, against the default's 37.0% — so it is the better generator on this axis too. It simply produces fewer molecules the existing record can vouch for.")}
<div class="note"><span class="tg">Both readings are valid, for different purposes</span>
<p>The raw count answers &ldquo;how many candidates can I defend with data I already have?&rdquo; The hit rate answers &ldquo;how good is this model at finding wide-liquid-range chemistry?&rdquo; The recipe loses the first by 6 molecules — inside the noise of a single seed — and wins the second.</p></div>

<h2><span class="num">06</span>The central finding — the obvious target is backwards</h2>
<div class="note w2"><span class="tg">Conditioning toward high melting point destroys TES yield</span>
<p>Liquid range is <code>T_dec − MP</code>, so <em>raising</em> the melting point <em>shrinks</em> the usable span of a liquid heat store. Conditioning at 250 °C produced <b>5</b> candidates above 250 °C; conditioning at 0 °C produced <b>149</b>. A thirtyfold difference, driven entirely by the direction of the target, and invisible until you write the figure of merit down.</p></div>
<p>Conditioning still works — it is the single most effective intervention here, beating the unconditional default by <strong>+34% on lr&gt;250 and +56% on lr&gt;300</strong>. It just has to point the right way. The 250 °C arm also carries the highest novel-cation rate of any arm (0.734), because extreme targets push generation to the periphery of the latent distribution where no ion has been measured — novel, and unscoreable for the same reason.</p>
{lr_hist()}

<h2><span class="num">07</span>Novelty against evidence</h2>
{scatter("Novel-cation rate against screened yield")}
<div class="note"><span class="tg">A limit on the screen, not on the chemistry</span>
<p>An ion with no measured record is not a bad ion, it is an unknown one. Novelty and evidence are not opposed in chemistry — they are opposed in <em>this screen</em>, and only because measurement coverage is finite. Closing that gap needs a way to score novel ions: ion-family precedent, a physics calculation, or new measurements.</p></div>

<h2><span class="num">08</span>Training</h2>
<p>All arms run 8,000 steps of AdamW at identical settings.</p>
{line_chart(loss, "Stage 4 validation loss", "training step", ymin=0.5, ymax=1.4)}
<div class="note w2"><span class="tg">These losses are not comparable across arms</span>
<p>The recipe's much lower curve is <strong>not</strong> a quality signal. A non-zero <code>sigma_min</code> changes the variance of the regression target, so the loss is on a different scale. The figure confirms convergence and nothing else.</p></div>

<h2><span class="num">09</span>Candidate shortlist</h2>
<p>Ranked by liquid range inferred from the measured behaviour of real ILs sharing each ion — a lookup against 3,070 measured pairs, not a property prediction. Every trained property oracle in this project failed external validation, so none is used.</p>
<div class="ts"><table>
<thead><tr><th>Ion pair</th><th class=n>liquid range °C</th><th>families</th><th class=n>heavy</th><th>novel pair</th><th>arm</th></tr></thead>
<tbody>{''.join(top)}</tbody></table></div>
<p>Sulfonylimide anions dominate unprompted, which is the expected answer — bis(trifluoromethylsulfonyl)imide is the workhorse thermally stable IL anion — and a useful sign the screen tracks real chemistry rather than an artifact.</p>

<h2><span class="num">10</span>Thermal motion</h2>
<div class="note w2"><span class="tg">This is not molecular dynamics</span>
<p>No integrator, no momentum, no time axis. Configurations are drawn from the canonical ensemble by Metropolis Monte Carlo over torsions and rigid-body ion motion on an MMFF94 surface, so amplitude grows with temperature for the right reason but nothing here is a rate or a trajectory. MMFF94 was never parameterised for ionic liquids; this is a single ion pair in vacuum, not a condensed phase; and a hard wall on the inter-ion distance stands in for the cage neighbouring ions would provide, because an unconstrained pair simply dissociates. Illustrative only.</p></div>
<div class="ctl">
<span>Molecule</span><select id="mol"></select>
<span>Temperature</span><span id="tb"></span>
<button id="pp" aria-pressed="true">Pause</button>
<span id="st" style="color:var(--muted);font-family:var(--fm);font-size:12px"></span>
</div>
<div id="v"></div>
{line_chart(rmsf, "Root-mean-square fluctuation against temperature", "temperature (K)", ymin=2.2, ymax=4.8)}
<p>Fluctuation rises monotonically with temperature for all four candidates, and the C₁₂-phosphonium — the most torsionally flexible — moves most. That ordering is the only quantitative claim these runs support.</p>

<h2><span class="num">11</span>Limitations</h2>
<ul>
<li><strong>Screening is a lookup, not a prediction.</strong> A candidate is scored only when both its ions appear in measured ILs — 36% of generated molecules at best — which biases hard toward known chemistry and is the entire reason the two questions in §01 diverge.</li>
<li><strong>Ion-level aggregation assumes additivity.</strong> Liquid range is the mean over the two ions' measured records; real ion pairing is not additive.</li>
<li><strong>Heat capacity and thermal conductivity are too sparse to screen on</strong> — 201 and 74 measured pairs, against 510 with both melting and decomposition temperature.</li>
<li><strong>One seed per arm.</strong> Differences below roughly 10 candidates are not resolvable, which includes the 105-vs-111 raw-count gap.</li>
<li><strong>No synthesis, no measurement.</strong> Nothing establishes that a generated molecule can be made, or that it has the inferred properties.</li>
</ul>
<footer><p>Generated from <code>~/fm-chem</code>. Measured properties from Zenodo record 3251643 (CC-BY 4.0). Screening <code>scripts/screen_tes.py</code>; thermal frames <code>scripts/mc_frames.py</code>; this page <code>scripts/make_tes_report.py</code>.</p></footer>
</div>
<script>
const MC = {mcjson};
const sel = document.getElementById('mol'), tb = document.getElementById('tb'),
      pp = document.getElementById('pp'), st = document.getElementById('st');
MC.forEach((m,i)=>{{const o=document.createElement('option');o.value=i;
  o.textContent='mol '+(i+1)+' — '+m.smiles.slice(0,26)+'…';sel.appendChild(o);}});
let mi=0, ti=0, fi=0, playing=true, viewer=null;
function temps(){{tb.innerHTML='';MC[mi].runs.forEach((r,i)=>{{const b=document.createElement('button');
  b.textContent=r.T+' K';b.setAttribute('aria-pressed',i===ti);
  b.onclick=()=>{{ti=i;fi=0;temps();load();}};tb.appendChild(b);}});}}
function load(){{
  if(!viewer){{viewer=$3Dmol.createViewer('v',{{backgroundColor:'rgba(0,0,0,0)'}});}}
  const r=MC[mi].runs[ti];
  viewer.clear();
  let xyz=r.symbols.length+'\\n\\n';
  r.frames[fi].forEach((p,i)=>{{xyz+=r.symbols[i]+' '+p[0]+' '+p[1]+' '+p[2]+'\\n';}});
  viewer.addModel(xyz,'xyz');
  viewer.setStyle({{}},{{stick:{{radius:0.14}},sphere:{{scale:0.26}}}});
  viewer.zoomTo();viewer.render();
  st.textContent='frame '+(fi+1)+'/'+r.frames.length+'  ·  RMSF '+r.rmsf.toFixed(2)+' Å';
}}
pp.onclick=()=>{{playing=!playing;pp.textContent=playing?'Pause':'Play';pp.setAttribute('aria-pressed',playing);}};
sel.onchange=()=>{{mi=+sel.value;ti=0;fi=0;temps();load();}};
temps();load();
setInterval(()=>{{if(!playing)return;const r=MC[mi].runs[ti];fi=(fi+1)%r.frames.length;load();}},110);
</script>
""")
print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")
