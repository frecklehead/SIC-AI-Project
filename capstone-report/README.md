# SIC-AI Capstone — Mid-Term Progress Report (LaTeX)

## Use on Overleaf
1. Overleaf → **New Project → Upload Project** → upload `report.zip`.
2. Menu → Compiler: **pdfLaTeX**, Main document: `main.tex`. Recompile.

## Build locally (Garuda Linux + VS Code)
```bash
# 1. TeX Live (Arch/Garuda package names)
sudo pacman -S --needed texlive-basic texlive-binextra texlive-latex texlive-latexrecommended \
  texlive-latexextra texlive-fontsrecommended texlive-bibtexextra texlive-pictures texlive-mathscience perl

# 2. (optional) only for regenerating plots
sudo pacman -S --needed python-matplotlib python-pandas python-numpy

# 3. Build
cd report
latexmk -pdf main.tex      # output: main.pdf (or build/main.pdf from VS Code)
latexmk -c                 # clean aux files
```
VS Code: install the **LaTeX Workshop** extension (James-Yu.latex-workshop), open the folder,
open `main.tex`, press **Ctrl+Alt+B** to build and **Ctrl+Alt+V** to preview the PDF in a tab.
Ctrl+click in the PDF jumps to the source line (SyncTeX). The build runs on save (see `.vscode/settings.json`).

If you use the open-source `code` build and cannot find the extension, search Open VSX for "LaTeX Workshop".
A missing `.sty` error means a texlive group is missing: `pacman -F <file>.sty` shows the package.

## Before submitting
- Search for `TODO` and resolve every item, then set `\showtodofalse` in `main.tex`.
- Put the TU logo at `figures/tu_logo.png` (optional).
