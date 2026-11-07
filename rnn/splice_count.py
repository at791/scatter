"""Swap the stream network + AI block of scatter_pfc.html for the serial-counting network and its live panels."""
import io, json
L = io.open('scatter_pfc.html', encoding='utf-8').read().split('\n')
note = next(i for i, l in enumerate(L) if 'id="aiNote"' in l)   # the status note moves from the dock to the playback bar under the board
assert L[note].strip() == '<span id="aiNote" aria-live="polite"></span>'
del L[note]
land =json.load(io.open('../mante/count_land.json', encoding='utf-8')); land = {k: land[k] for k in ('mu', 'P')}   # the page only needs the plane itself
r3 = json.load(io.open('../mante/count_rib3d.json', encoding='utf-8'))
for v in r3['nums'].values():
    for k in ('q', 'rd', 'ar', 'view'): v.pop(k, None)                                                        # the map draws geometry only
js = io.open('count_ai.js', encoding='utf-8').read().replace('@@WEIGHTS@@', io.open('../mante/count_b64.txt').read().strip())
js = js.replace('@@LAND@@', json.dumps(land, separators=(',', ':')))
js = js.replace('@@TROUGH@@', io.open('../mante/count_trough.json', encoding='utf-8').read().strip())
js = js.replace('@@CROSS@@', io.open('../mante/count_cross.json', encoding='utf-8').read().strip())
js = js.replace('@@RIB3D@@', json.dumps(r3, separators=(',', ':'))).rstrip('\n').split('\n')
assert not any('@@' in l for l in js)
s = next(i for i, l in enumerate(L) if 'PFC-style recurrent network' in l)
e = next(i for i, l in enumerate(L) if i > s and l.strip() == '})();')
assert 'const MN_WEIGHTS' in '\n'.join(L[s:e])
L[s:e] = js
stage = next(i for i, l in enumerate(L) if 'id="busy"' in l) + 1
assert L[stage].strip() == '</div>'
L[stage + 1:stage + 1] = [
    '  <div class="aiplay" id="aiPlay" hidden><span id="aiNote" aria-live="polite"></span></div>',
    '  <section class="aiviz" id="aiViz" hidden aria-label="What the network is doing">',
    '    <p class="note" id="vzNote">&nbsp;</p>',
    '    <div class="panels">',
    '      <svg id="vz3" viewBox="0 0 300 300" role="img" aria-label="Map of the troughs of all 15 numbers in 3-D from one fixed view: this circle\'s trough in bold, earlier circles\' paths faded, the state moving on it; drag to rotate"></svg>',
    '      <svg id="vzT" viewBox="0 0 300 300" role="img" aria-label="This number\'s trough unrolled (counting across) against the direction toward the answers, with the three verdict regions and the state moving on it"></svg>',
    '      <svg id="vzL" viewBox="0 0 300 300" role="img" aria-label="Cross-section of this number\'s trough: the trace hidden stones leave across, the push of dark stones up, with the speed and flow of the network there and the state moving on it"></svg>',
    '      <svg id="vzI" viewBox="0 0 300 300" role="img" aria-label="What the network sees: its five input channels through the current circle"></svg>',
    '      <svg id="vzO" viewBox="0 0 300 300" role="img" aria-label="The three output units through the current circle"></svg>',
    '      <svg id="vzP" viewBox="0 0 300 300" role="img" aria-label="Real population plane: mean activity of the need-type and slack-type units"></svg>',
    '    </div>',
    '  </section>']
css = next(i for i, l in enumerate(L) if l.startswith('#aiNote{'))
L[css:css + 1] = [
    '#aiNote{font-size:.85rem;color:var(--muted);min-width:11em;text-align:left}',
    '.aiplay{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;justify-content:center}',
    '.aiplay .aipos{font-size:.85rem;color:var(--muted);width:7.5em;white-space:nowrap;text-align:center;font-variant-numeric:tabular-nums}',
    '.aiplay #aiNote{flex-basis:100%;text-align:center;min-width:0;min-height:1.3em}',   # its own line, so its changing length does not move the buttons
    '#aiCtl .btn{justify-content:center}',
    '.btn:disabled{opacity:.4;cursor:default}',
    '#board .ai .scan{stroke:var(--fg);stroke-opacity:.45;stroke-dasharray:6 6;stroke-width:2.5}',
    '#board .ai .fov{stroke:#d85a30;stroke-width:5}',
    '#board .ai .seen{fill:var(--fg);fill-opacity:.09;stroke:none}',
    '#board .ai .now{fill:var(--accent);stroke:none}',
    '.aiviz{max-width:980px;margin:6px auto 0;width:100%}',
    '.aiviz .note{margin:0 0 4px;text-align:center;color:var(--muted);font-size:.85rem;min-height:1.4em}',
    '.aiviz .note b{color:var(--fg)}',
    '.aiviz .panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px}',
    '.aiviz svg{width:100%;height:auto;display:block;font-family:var(--font)}',
    '.aiviz .th{font-size:12px;font-weight:600;fill:var(--fg)}',
    '.aiviz .ts{font-size:9.5px;fill:var(--muted)}',
    '.aiviz .grid{stroke:var(--line);stroke-width:.5}']
assert '#board .ai .pick' in '\n'.join(L[:css])
if not any('<meta charset' in l for l in L):
    t = next(i for i, l in enumerate(L) if '<title>' in l); L[t:t] = ['<meta charset="utf-8">']
io.open('scatter_pfc.html', 'w', encoding='utf-8', newline='\n').write('\n'.join(L))
print('ok', len(L), 'lines; weights', len(js))
