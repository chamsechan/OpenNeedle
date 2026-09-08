#!/usr/bin/env python3
"""Generate self-contained README SVG artwork from the recorded benchmark."""
import hashlib
from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/backend_comparison.json'
OUT = ROOT / 'docs/assets'

PALETTES = {
    'light': dict(bg='#f6f7f3', panel='#ffffff', ink='#182724', muted='#53675f',
                  border='#d6dfd7', grid='#c9d8ce', accent='#087d68', soft='#dceee5',
                  teal='#2e9a85', official='#53625c', torch='#9ba69f', track='#e7ece6',
                  core='#142e27', coreink='#eafff2', coreline='#386454'),
    'dark': dict(bg='#0c1815', panel='#11231d', ink='#edf6ef', muted='#a3b8ac',
                 border='#294437', grid='#294e3c', accent='#7cebb7', soft='#203e30',
                 teal='#3ab28e', official='#a8b9ae', torch='#667e70', track='#20382c',
                 core='#17382b', coreink='#eafff2', coreline='#508668'),
}


def draw(theme, report, static=False):
    c = PALETTES[theme]
    summary = report['summary']
    ratio = summary['native_sdot']['decode_tps_median'] / summary['pytorch_t2']['decode_tps_median']
    rows = [('official', 'Official engine', 'official'),
            ('native_sdot', 'OpenNeedle / SDOT*', 'accent'),
            ('native_fp32', 'OpenNeedle / FP32', 'teal'),
            ('pytorch_t2', 'PyTorch / FP32', 'torch')]
    maximum = max(summary[name]['decode_tps_median'] for name, _, _ in rows)
    pieces = []
    def add(value): pieces.append(value)
    def text(x, y, value, size=14, color='ink', weight=400, extra=''):
        add(f'<text x="{x}" y="{y}" fill="{c.get(color,color)}" font-size="{size}" font-weight="{weight}" {extra}>{escape(str(value))}</text>')
    def rect(x,y,w,h,fill,rx=0,stroke=None,extra=''):
        add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{c.get(fill,fill)}"'+
            (f' stroke="{c.get(stroke,stroke)}"' if stroke else '')+f' {extra}/>')
    def path(d,color='border',width=1,extra=''):
        add(f'<path d="{d}" fill="none" stroke="{c.get(color,color)}" stroke-width="{width}" {extra}/>')
    add('<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="500" viewBox="0 0 1200 500" role="img" aria-labelledby="title desc">')
    add('<title id="title">OpenNeedle — packed CPU inference and measured decode throughput</title>')
    add('<desc id="desc">Conceptual packed GEMV data flow. On a 4-core ARM Neoverse-N1 CPU: official '+
        f"{summary['official']['decode_tps_median']:.2f}, OpenNeedle SDOT {summary['native_sdot']['decode_tps_median']:.2f}, "+
        f"OpenNeedle FP32 {summary['native_fp32']['decode_tps_median']:.2f}, and PyTorch {summary['pytorch_t2']['decode_tps_median']:.2f} tokens per second. "+
        'Medians of three queries repeated five times. Application timing definitions differ; SDOT adds quantization error. Motion is illustrative.</desc>')
    add('<metadata>'+escape(json.dumps({'source':'reports/backend_comparison.json','sha256':hashlib.sha256(REPORT.read_bytes()).hexdigest(),'theme':theme,'static':static}))+'</metadata>')
    add('<style>text{font-family:Inter,"Segoe UI",Arial,sans-serif}.mono{font-family:"SFMono-Regular",Consolas,"Liberation Mono",monospace}.overline{letter-spacing:2.2px}.flow{stroke-dasharray:5 18;stroke-linecap:round;animation:travel 2.6s linear infinite}.phase2{animation-delay:-1.3s}.phase3{animation-duration:1.8s}.spark{animation:shimmer 3.4s ease-in-out infinite}.spark2{animation-delay:-1.7s}@keyframes travel{to{stroke-dashoffset:-92}}@keyframes shimmer{0%,100%{opacity:.38}50%{opacity:1}}@media(prefers-reduced-motion:reduce){.flow,.spark{animation:none!important}.flow{stroke-dasharray:5 18;opacity:.7}}'+
        ('.flow,.spark{animation:none!important}' if static else '')+'</style>')
    add('<defs><pattern id="grid" width="24" height="24" patternUnits="userSpaceOnUse">'+
        f'<circle cx="1" cy="1" r=".7" fill="{c["grid"]}"/></pattern></defs>')
    rect(.5,.5,1199,499,'bg',20,'border')
    rect(18,18,676,464,'url(#grid)',14,extra='opacity=".42"')
    rect(715,18,467,464,'panel',14)
    path('M700 36V464')
    # A small stitched monogram echoes a needle, rather than a generic chip logo.
    path('M55 46V28L69 46V28M80 28V46M76 37H84','accent',2,extra='stroke-linecap="round" stroke-linejoin="round"')
    text(99,41,'OPEN COMPUTE / CQ2.2',11,'muted',600,'class="overline"')
    text(50,104,'OpenNeedle',53,'ink',650,'letter-spacing="-2"')
    text(53,134,'PyTorch conversion + packed CPU inference',16,'muted')
    # One projection: weights feed the packed dot; activations enter from above.
    text(53,187,'COMPRESSED WEIGHTS',11,'muted',600,'class="overline"')
    rect(267,164,148,36,'soft',18)
    text(341,187,'activation x',13,'accent',550,'text-anchor="middle" class="mono"')
    path('M341 200V226','border',2)
    path('M341 200V226','accent',2,extra='class="flow phase3"')
    rect(52,217,123,107,'panel',12,'border')
    for row in range(5):
        for col in range(7):
            opacity=[.23,.43,.68,1][(row*7+col*3)%4]
            rect(64+col*14,230+row*16,9,10,'accent',2,
                 extra=f'opacity="{opacity}"'+(' class="spark"' if (row,col)==(2,4) else ' class="spark spark2"' if (row,col)==(3,1) else ''))
    text(52,349,'CQ2 / CQ4',15,'ink',600)
    text(52,369,'codes + FP16 norms',12,'muted')
    path('M175 272H268','border',2)
    path('M175 272H268','accent',2.5,extra='class="flow"')
    # Core and PCB pins are static; only small packets move along the buses.
    for y in range(241,315,12):
        path(f'M258 {y}H268M416 {y}H426','coreline',2)
    for x in range(287,405,13):
        path(f'M{x} 216V226M{x} 324V334','coreline',2)
    rect(268,226,148,98,'core',12,'coreline')
    rect(278,236,128,78,'none',7,'coreline')
    text(342,254,'PACKED',10,'coreink',500,'text-anchor="middle" letter-spacing="2.5"')
    text(342,283,'GEMV',28,'coreink',600,'text-anchor="middle" class="mono"')
    text(342,304,'Hx + CQ lookup',11,'coreink',400,'text-anchor="middle" class="mono"')
    text(342,350,'NEON / SDOT',15,'ink',600,'text-anchor="middle"')
    text(342,369,'shared input transform',12,'muted',400,'text-anchor="middle"')
    path('M416 272H520','border',2)
    path('M416 272H520','accent',2.5,extra='class="flow phase2"')
    rect(520,229,125,95,'panel',12,'border')
    text(536,253,'OUTPUT',10,'muted',500,'class="overline"')
    for y,width in [(265,81),(278,48),(291,65)]:
        rect(536,y,93,5,'track',2)
        rect(536,y,width,5,'accent' if y==278 else 'teal',2)
    text(520,349,'token logits',15,'ink',600)
    text(520,369,'grammar → token',12,'muted')
    path('M52 391H652')
    for x,value,label in [(52,'13.74 MB','DEPLOYMENT FILE'),(265,'43.6M','DEPLOYED PARAMETERS'),(510,'FP32 KV','CURRENT CACHE')]:
        text(x,430,value,26,'ink',550,'letter-spacing="-.5"')
        text(x,451,label,10,'muted',500,'letter-spacing="1.3"')
    text(52,477,'One projection, illustrated. Packet motion is not a timing measurement.',11,'muted')
    # Truthful comparison bars: same origin and strictly linear, untruncated scale.
    text(751,48,'MEASURED ON CPU',11,'muted',600,'class="overline"')
    text(751,80,'Decode throughput',23,'ink',600,'letter-spacing="-.4"')
    text(751,103,'4-core ARM Neoverse-N1 · median token/s',12,'muted')
    for index,(name,label,color) in enumerate(rows):
        y=145+index*69
        value=summary[name]['decode_tps_median']
        text(751,y,label,14,'ink',550)
        text(1145,y+2,f'{value:.2f}',25,color,600,'text-anchor="end" class="mono" letter-spacing="-.8"')
        rect(751,y+15,394,5,'track',2)
        rect(751,y+15,round(394*value/maximum,3),5,color,2,
             extra=f'data-backend="{name}" data-value="{value:.12g}"')
    path('M751 391H1145')
    text(751,434,f'{ratio:.1f}×',36,'accent',600,'letter-spacing="-1.5"')
    text(890,420,'SDOT / PyTorch FP32',14,'ink',500)
    text(890,442,'Same model · application comparison',10.5,'muted')
    text(751,467,'* SDOT adds INT8 rounding. Timing definitions differ.',11,'muted')
    text(751,484,'Source: recorded benchmark · 3 queries × 5 repetitions',10,'muted')
    add('</svg>')
    return '\n'.join(pieces)+'\n'


def main():
    report=json.loads(REPORT.read_text())
    OUT.mkdir(parents=True,exist_ok=True)
    for theme in PALETTES:
        path=OUT/f'openeedle-hero-{theme}.svg'
        path.write_text(draw(theme,report),encoding='utf-8')
        print(path.relative_to(ROOT))
    path=OUT/'openeedle-hero-static.svg'
    path.write_text(draw('light',report,static=True),encoding='utf-8')
    print(path.relative_to(ROOT))
    path=OUT/'openeedle-hero-static-dark.svg'
    path.write_text(draw('dark',report,static=True),encoding='utf-8')
    print(path.relative_to(ROOT))


if __name__=='__main__':
    main()
