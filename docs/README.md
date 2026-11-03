# Documents

- `midterm-progress-report.pdf` — compiled mid-term progress report (16 pages).
- `proposal.pdf` — original SIC-AI capstone proposal.
- `study-notes-and-viva-prep.docx` — verified architecture, VAD and viva preparation notes.

The report source is in [`../report/`](../report/). Rebuild it with:

```bash
cd report
latexmk -pdf -interaction=nonstopmode main.tex
cp main.pdf ../docs/midterm-progress-report.pdf
```
