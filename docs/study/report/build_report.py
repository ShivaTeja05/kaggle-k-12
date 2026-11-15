"""Build the research report: report.md -> report.html -> K12_RESEARCH_REPORT.pdf (headless Chrome).
Run from the repo root:  python docs/study/report/build_report.py"""
from pathlib import Path
import subprocess
import markdown

HERE = Path(__file__).parent
CSS = """
@page { size: A4; margin: 17mm 16mm 18mm 16mm; @bottom-center { content: counter(page); font: 9pt Helvetica, Arial, sans-serif; color: #52514e; } }
body { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 10.2pt; line-height: 1.5; color: #14171f; margin: 0; }
h1, h2, h3 { color: #12305f; break-after: avoid; }
h2 { font-size: 15pt; border-bottom: 1.5px solid #2a78d6; padding-bottom: 3px; margin-top: 26px; break-before: page; }
h2:first-of-type { break-before: avoid; }\nh2[id^="appendix-b"], h2[id^="appendix-d"] { break-before: auto; }
h3 { font-size: 11.8pt; margin-top: 18px; }
p, li { orphans: 3; widows: 3; }
.titlepage { text-align: center; padding-top: 70mm; break-after: page; }
.titlepage .title { font-size: 25pt; border: 0; margin: 0 0 8px 0; color: #12305f; }
.titlepage .subtitle { font-size: 14pt; color: #33415c; margin: 0 0 26px 0; }
.titlepage .meta { font-size: 10pt; color: #52514e; }
.abstract { border-left: 3px solid #2a78d6; background: #f4f7fc; padding: 8px 14px; margin: 0 0 8px 0; font-size: 10pt; }
table { border-collapse: collapse; margin: 10px 0 14px 0; font-size: 8.9pt; width: 100%; }
th, td { border: 1px solid #cdd5e0; padding: 4px 6px; vertical-align: top; text-align: left; }
th { background: #eaf1fb; }
tr { break-inside: avoid; }
code { background: #f1f3f7; padding: 0 3px; border-radius: 3px; font-size: 8.8pt; }
figure { margin: 12px 0 16px 0; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; max-height: 150mm; }
figure img.small { max-width: 62%; max-height: 95mm; }
figure img.tall { max-height: 228mm; }
figcaption { font-size: 8.8pt; color: #52514e; margin-top: 5px; text-align: left; }
.eq { text-align: center; font-size: 11pt; margin: 10px 0; }
.toc ul { list-style: none; padding-left: 14px; margin: 2px 0; } .toc > ul { padding-left: 0; } .toc a { color: #12305f; text-decoration: none; }
"""
md = markdown.Markdown(extensions=['tables', 'toc', 'sane_lists', 'md_in_html'], extension_configs={'toc': {'toc_depth': '2-3'}})
import re
src = (HERE / 'report.md').read_text()
# Number figures in order of appearance and repoint in-text references written with the draft numbers.
labels = re.findall(r'<figcaption>Figure (\w+)\.', src)
new = {old: str(i + 1) for i, old in enumerate(labels)}
src = re.sub(r'Figure (\w+)\b', lambda m: 'Figure ' + new.get(m.group(1), m.group(1)), src)
body = md.convert(src)
html = f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>K-12 Misconception Detection: research report</title><style>{CSS}</style></head><body>{body}</body></html>'
(HERE / 'report.html').write_text(html)
pdf = HERE.parent / 'K12_RESEARCH_REPORT.pdf'
chrome = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
subprocess.run([chrome, '--headless=new', '--disable-gpu', '--no-pdf-header-footer', f'--print-to-pdf={pdf}', (HERE / 'report.html').as_uri()],
               check=True, capture_output=True, timeout=180)
print('wrote', pdf, pdf.stat().st_size // 1024, 'KB')
