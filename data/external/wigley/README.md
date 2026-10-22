# data/external/wigley -- Wigley hull measurements for Track B (Rankine solver, step B4)

Provenance, sha256 and table/page references of every file: `data/external/SOURCES.md`, section `wigley/`.

## Licence notices

- **`wavebem_*` files** (`wavebem_wigleyDrag*.csv`, `wavebem_tokyoFr*.csv`, `wavebem_tokyoWigley.pdf`) are unchanged
  copies (renamed with the prefix `wavebem_`) of the files in `experimental-results/wigley/` of WaveBEM
  (github.com/mathLab/WaveBEM, commit b44b0fb, 2017-03-07), Copyright (C) 2012-2016 Andrea Mola and Luca Heltai,
  distributed under the GNU Lesser General Public License version 2.1 or later
  (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html; full text in the WaveBEM repository, file `LICENSE`).
  The CSV files are the WaveBEM authors' digitisations of scanned figures; `wavebem_tokyoWigley.pdf` is a scan of
  Hendrix & Noblesse (1988), "Validation of a simple approximate method for evaluating steady ship waves" (David Taylor
  Research Center), pages 61-64 of a workshop volume, as shipped with WaveBEM. Known defect (B4 review, stage note
  `learn/meta/RANKINE_STAGE1_2026-10-06.md` section 9.7.1): the four `wavebem_wigleyDragExp*.csv` C_W series do not
  reproduce the measured markers of that scan at Fn < 0.3 (they follow the slender-ship theory lines); the fourth file
  is a third U. Tokyo series (UT3), not Y.N.U.
- **`yang_lohner_noblesse_huang_2000_eccomas367.pdf`** (Yang, Lohner, Noblesse & Huang 2000, ECCOMAS paper 367) is a
  copyrighted conference paper, kept locally only so that the transcription of its vector figures
  (`yang2000_*.csv`, `studies/rankine_wigley_data.py`) can be re-checked. **Do not include it in any published copy of
  this repository**: the extracted CSV files and the sha256 recorded in SOURCES.md suffice (it is excluded by the
  `.gitignore` of this folder).
- **`yang2000_*.csv`** and **`hn1988_p61_*.csv`** are our transcriptions (numbers only) of published figures, used for
  comparison: `yang2000_*` from the PDF drawing commands (`studies/rankine_wigley_data.py`), `hn1988_p61_*` read from
  the scan by `studies/rankine_wigley_scan.py` (stated calibration and reading error).
