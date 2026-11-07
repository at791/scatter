"""Rebuild scatter_pfc.html from the base game page.  python rebuild.py   (run inside rnn/)
1. scatter_base.html -> scatter_pfc.html with the "RNN" button labels;
2. the unit groups and reference ticks in count_ai.js are refreshed from ../mante/count_frames_w.json;
3. splice_count.py splices count_ai.js (weights from ../mante/count_b64.txt, panel data from ../mante/count_*.json) into the page.
Everything it reads is produced in ../mante (see README, "Rebuilding the page")."""
import io, json, re, subprocess, sys
s = io.open('scatter_base.html', encoding='utf-8').read()
for a, b in [('>Watch the AI play<', '>Watch the RNN play<'),
             ('title="Let the neural network play from here">AI plays<', 'title="Let the rate network play from here">RNN plays<')]:
    assert s.count(a) == 1, a
    s = s.replace(a, b)
io.open('scatter_pfc.html', 'w', encoding='utf-8', newline='\n').write(s)

F = json.load(io.open('../mante/count_frames_w.json', encoding='utf-8'))
js = io.open('count_ai.js', encoding='utf-8').read()
fmt = lambda v: '[' + ','.join(str(int(x)) for x in v) + ']'
ref = lambda d: '[' + ','.join(f'[{int(k)},{v}]' for k, v in sorted(d.items(), key=lambda kv: int(kv[0]))) + ']'
lines = {'CN_NEED': f'const CN_NEED={fmt(F["need_u"])};',
         'CN_NEED_S': f'const CN_NEED_S={fmt(F["sn"])};',
         'CN_SLACK': f'const CN_SLACK={fmt(F["slack_u"])},CN_SLACK_S={fmt(F["ss"])};',
         'CN_REF': 'const CN_REF={need:' + ref(F['ref']['need']) + ',slack:' + ref(F['ref']['slack']) + '};'}
for key, line in lines.items():
    pat = re.compile(r'^const ' + key + r'=.*?;', re.M)
    assert len(pat.findall(js)) == 1, key
    js = pat.sub(lambda m: line, js, count=1)
assert f'const CN_N={F["N"]},' in js, 'count_ai.js is built for a different network size'
io.open('count_ai.js', 'w', encoding='utf-8', newline='\n').write(js)

subprocess.run([sys.executable, 'splice_count.py'], check=True)
print('rebuilt scatter_pfc.html')
