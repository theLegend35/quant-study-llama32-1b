# Paper

`paper_draft.pdf` (and `paper_draft.docx`) is the current draft. `build_paper.py` is the source; all `tools/patch_*.py` patches have already been applied to this copy.

To rebuild (needs `python-docx`, matplotlib, and LibreOffice `soffice` for the PDF):

```bash
python3 paper/build_paper.py
```

The copy of `build_paper.py` in this folder uses paths relative to `paper/`. The `tools/patch_*.py` scripts still refer to the original drafting paths (`/workspace/paper/build_paper.py`) and are kept so the edit history is reproducible; you do not need to re-run them to build the current draft. Draft backups are not included.
