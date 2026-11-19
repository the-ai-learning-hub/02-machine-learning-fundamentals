"""Verificação de layout (ronda 2): tamanho mínimo de texto, sobreposições, área vazia, transbordo + folhas de contacto.
Uso: PWCHANNEL=msedge python test_layout.py [--shots]   (servidor em 127.0.0.1:8765)"""
import os, sys, json
from playwright.sync_api import sync_playwright
CH = os.environ.get('PWCHANNEL'); URL = 'http://127.0.0.1:8765/index.html'
OUT = 'C:/Users/diogo/Desktop/02-machine-learning-fundamentals/_planning/deck_tests/shots/all/'
NOTR = '*{transition:none!important;animation:none!important}.type{width:auto!important}.rv,pre.py .ln,.tl{opacity:1!important;transform:none!important}'
JS = r"""(skipFill) => {
  const s = document.querySelector('.slide.active'); const r = s.getBoundingClientRect(); const sc = r.width / 1280; const out = [];
  const main = s.querySelector('.main'); const mr = main.getBoundingClientRect();
  if (main.scrollHeight > main.clientHeight + 1) out.push('SCROLL main ' + main.scrollHeight + '>' + main.clientHeight);
  const EX = '.kicker,.rail,.pill,.label,.tag,.num,.rain,.notes,.sep,.cursor,.dot';
  const leaves = [];
  const walker = document.createTreeWalker(main, NodeFilter.SHOW_TEXT);
  let n;
  while ((n = walker.nextNode())) {
    const t = n.textContent.trim(); if (!t) continue; const el = n.parentElement;
    if (el.closest(EX) || el.closest('defs,style,script')) continue;
    const cs = getComputedStyle(el); if (cs.visibility === 'hidden' || cs.display === 'none') continue;
    let size, rect, rects;
    if (el.closest('svg')) { const m = el.getScreenCTM && el.getScreenCTM(); if (!m) continue; size = parseFloat(cs.fontSize) * m.a / sc; const b = el.getBoundingClientRect(); rect = b; rects = [b]; }
    else { size = parseFloat(cs.fontSize); const rg = document.createRange(); rg.selectNodeContents(n); const b = rg.getBoundingClientRect(); rect = b; rects = Array.from(rg.getClientRects()); }
    if (!rect.width || !rect.height) continue;
    if (cs.opacity === '0') continue;
    leaves.push({t: t.slice(0, 24), size, rect, rects, el, svg: !!el.closest('svg')});
  }
  const lim = 14.9;
  const small = {}; leaves.forEach(l => { if (l.size < lim) small[l.t] = l.size.toFixed(1); });
  Object.entries(small).slice(0, 4).forEach(([t, v]) => out.push('SMALL ' + v + ' "' + t + '"'));
  // clip: text outside slide
  leaves.forEach(l => { const b = l.rect; if (b.right > r.right + 2 || b.bottom > r.bottom + 2 || b.left < r.left - 2 || b.top < r.top - 2) out.push('OUT "' + l.t + '"'); });
  // overlaps
  let ov = 0;
  for (let i = 0; i < leaves.length && ov < 4; i++) for (let j = i + 1; j < leaves.length && ov < 4; j++) {
    const a = leaves[i], b = leaves[j]; if (a.el === b.el) continue;
    let hit = false;
    for (const ra of a.rects) for (const rb of b.rects) { const x = Math.min(ra.right, rb.right) - Math.max(ra.left, rb.left), y = Math.min(ra.bottom, rb.bottom) - Math.max(ra.top, rb.top); if (x > 3 * sc && y > 0.35 * Math.min(ra.height, rb.height)) hit = true; }
    if (hit) { out.push('OVERLAP "' + a.t + '" / "' + b.t + '"'); ov++; }
  }
  // fill
  if (!skipFill) {
    let top = 1e9, bot = -1e9;
    main.querySelectorAll('svg,canvas,table,.card,.call,.simple,.disc,.term,.diagbox,li,p,h2,h1,.cm,.stat,.btn,.status').forEach(e => { if (e.closest('.rain')) return; const b = e.getBoundingClientRect(); if (!b.height) return; top = Math.min(top, b.top); bot = Math.max(bot, b.bottom); });
    const useful = (mr.height - 72 * sc);
    const frac = (bot - top) / useful;
    if (frac < 0.75) out.push('EMPTY ' + (frac * 100).toFixed(0) + '%');
    // empty bottom: free space below last content
    const free = (mr.bottom - 30 * sc) - bot; if (free > 120 * sc) out.push('GAPBOTTOM ' + Math.round(free / sc) + 'px');
  }
  return out;
}"""
def main():
    shots = '--shots' in sys.argv
    os.makedirs(OUT, exist_ok=True)
    errs = []; bad = []
    with sync_playwright() as p:
        b = p.chromium.launch(channel=CH) if CH else p.chromium.launch()
        pg = b.new_context(viewport={'width': 1280, 'height': 720}).new_page()
        pg.on('console', lambda m: errs.append(m.text) if m.type == 'error' else None); pg.on('pageerror', lambda e: errs.append(str(e)))
        pg.goto(URL); pg.add_style_tag(content=NOTR); pg.wait_for_timeout(600)
        n = pg.evaluate('document.querySelectorAll(".slide").length')
        for i in range(n):
            pg.evaluate(f'location.hash="#{i+1}"'); pg.wait_for_timeout(110)
            div = pg.evaluate('document.querySelector(".slide.active").classList.contains("div") || document.querySelector(".slide.active").classList.contains("cover-slide")')
            res = pg.evaluate(JS, div)
            if res: bad.append((i + 1, res))
            if shots: pg.screenshot(path=f'{OUT}{i+1:03d}.png')
        b.close()
    for i, r in bad: print(i, ' | '.join(r)[:400])
    print('slides', n, 'with issues', len(bad), 'console errors', errs[:5])
    if shots:
        from PIL import Image
        files = sorted(f for f in os.listdir(OUT) if f.endswith('.png') and f[0].isdigit())
        for k in range(0, len(files), 6):
            ims = [Image.open(OUT + f).resize((960, 540)) for f in files[k:k + 6]]
            sheet = Image.new('RGB', (1920, 1620), 'black')
            for q, im in enumerate(ims): sheet.paste(im, ((q % 2) * 960, (q // 2) * 540))
            sheet.save(f'{OUT}sheet_{k // 6 + 1:02d}.jpg', quality=80)
main()
