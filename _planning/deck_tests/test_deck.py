import sys, json, time
from playwright.sync_api import sync_playwright
import os
CH = os.environ.get('PWCHANNEL')
URL = 'http://127.0.0.1:8765/index.html'
SHOTS = 'C:/Users/diogo/Desktop/02-machine-learning-fundamentals/_planning/deck_tests/shots/'
NOTR = '*{transition:none!important;animation:none!important}'
JS_CHECK = """() => {
  const s = document.querySelector('.slide.active'); const r = s.getBoundingClientRect(); const out = [];
  const m = s.querySelector('.main');
  if (m.scrollHeight > m.clientHeight + 1) out.push('main scrollH ' + m.scrollHeight + '>' + m.clientHeight);
  const sc = r.width / 1280;
  s.querySelectorAll('.main *').forEach(e => {
    if (e.closest('canvas,.notes,svg defs') || e.tagName==='canvas') return;
    const b = e.getBoundingClientRect(); if (!b.width || !b.height) return;
    const mr = m.getBoundingClientRect();
    if (b.right > r.right + 2 || b.bottom > r.bottom + 2 || b.left < r.left - 2 || b.top < r.top - 2) out.push('out ' + e.tagName + '.' + (e.getAttribute('class')||'') + ' ' + Math.round((b.right-r.right)/sc) + ',' + Math.round((b.bottom-r.bottom)/sc));
  });
  s.querySelectorAll('svg text').forEach(t => { const m = t.getScreenCTM(); if (!m) return; const fs = parseFloat(getComputedStyle(t).fontSize) * m.a / sc; if (fs < 14.4 && t.textContent.trim()) out.push('smalltext ' + fs.toFixed(1) + ' ' + t.textContent.slice(0,20)); });
  return out.slice(0,6);
}"""
def run(vw, vh, tag, reduced=False, shots=()):
    errs = []; fails = []
    with sync_playwright() as p:
        b = p.chromium.launch(channel=CH) if CH else p.chromium.launch()
        ctx = b.new_context(viewport={'width': vw, 'height': vh}, reduced_motion='reduce' if reduced else 'no-preference')
        pg = ctx.new_page()
        pg.on('console', lambda m: errs.append(m.text) if m.type == 'error' else None)
        pg.on('pageerror', lambda e: errs.append('PAGEERR ' + str(e)))
        pg.on('requestfailed', lambda r: fails.append(r.url))
        pg.goto(URL); pg.add_style_tag(content=NOTR); pg.wait_for_timeout(500)
        n = pg.evaluate('document.querySelectorAll(".slide").length'); print(tag, 'slides', n)
        bad = []
        for i in range(n):
            pg.evaluate(f'location.hash="#{i+1}"'); pg.wait_for_timeout(120)
            a = pg.evaluate('document.querySelector(".slide.active")===document.querySelectorAll(".slide")[%d]' % i)
            res = pg.evaluate(JS_CHECK)
            if res or not a: bad.append((i + 1, a, res))
            if i + 1 in shots:
                pg.wait_for_timeout(400); pg.screenshot(path=f'{SHOTS}{tag}_{i+1:02d}.png')
        for x in bad: print('OVERFLOW', x)
        print('console errors', errs[:8]); print('failed requests', [f for f in fails if 'fonts.g' not in f][:5], 'fonts failed', len([f for f in fails if 'fonts.g' in f]))
        b.close()
if __name__ == '__main__':
    shots = [int(x) for x in sys.argv[4].split(',')] if len(sys.argv) > 4 else ()
    run(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], shots=shots)
