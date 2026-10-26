"""Render the repository's Markdown documents to PDF (python-markdown -> HTML -> headless Chromium via Node Playwright).

Usage: python3 scripts/render_pdf.py docs/op-print-design-v1.3.md [more.md ...]   (writes <name>.pdf next to each file)
Needs: pip install markdown; node with the `playwright` package and its Chromium.
Patterns such as `[j∈U](q_j − p_j)` would render as links; keep them out of the Markdown (see v1.2 §7).
"""
import json, pathlib, re, subprocess, sys, tempfile
import markdown

CSS = """
body { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 10pt; line-height: 1.45; color: #111; margin: 0; }
h1 { font-size: 16pt; } h2 { font-size: 13pt; border-bottom: 1px solid #bbb; padding-bottom: 2px; } h3 { font-size: 11pt; }
table { border-collapse: collapse; margin: 6px 0; font-size: 8.5pt; } th, td { border: 1px solid #999; padding: 2px 5px; vertical-align: top; }
th { background: #eee; } code, pre { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.5pt; }
pre { background: #f5f5f5; padding: 6px; white-space: pre-wrap; } blockquote { color: #444; border-left: 3px solid #ccc; margin-left: 0; padding-left: 8px; }
"""


NODE = """
const { chromium } = require('playwright');
(async () => {
  const jobs = JSON.parse(process.argv[1]);
  const browser = await chromium.launch();
  const page = await browser.newPage();
  for (const [html, pdf] of jobs) {
    await page.goto('file://' + html);
    await page.pdf({ path: pdf, format: 'A4', printBackground: true,
                     margin: { top: '16mm', bottom: '16mm', left: '14mm', right: '14mm' } });
    console.log('wrote ' + pdf);
  }
  await browser.close();
})();
"""


ITEM = re.compile(r"^( *)([-*]|\d+\.) ")


def normalize_lists(text):
    """python-markdown needs 4 spaces per nesting level; the docs use 2 (under '-') or 3 (under '1.')."""
    out, stack = [], []                      # stack of (source indent of an item, its content column)
    for line in text.split("\n"):
        m = ITEM.match(line)
        ind = len(line) - len(line.lstrip(" "))
        if not line.strip():
            out.append(line); continue
        if m:
            while stack and stack[-1][0] >= ind:
                stack.pop()
            stack.append((ind, ind + len(m.group(2)) + 1))
            out.append(" " * 4 * (len(stack) - 1) + line.lstrip(" "))
        elif stack and ind > 0:
            while stack and stack[-1][1] > ind:  # continuation paragraph: attach to the deepest item it fits under
                stack.pop()
            out.append(" " * 4 * len(stack) + line.lstrip(" ") if stack else line)
        else:
            stack = []
            out.append(line)
    return "\n".join(out)


def to_html(md_path, tmpdir):
    text = pathlib.Path(md_path).read_text()
    if re.search(r"\]\(", text):
        print(f"warning: {md_path} contains ']('; check that no formula renders as a link")
    body = markdown.markdown(normalize_lists(text), extensions=["tables", "fenced_code", "sane_lists"])
    html = pathlib.Path(tmpdir) / (pathlib.Path(md_path).stem + ".html")
    html.write_text(f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>")
    return str(html), str(pathlib.Path(md_path).with_suffix(".pdf").resolve())


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        jobs = [to_html(f, tmp) for f in sys.argv[1:]]
        subprocess.run(["node", "-e", NODE, json.dumps(jobs)], check=True)
