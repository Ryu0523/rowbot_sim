# wigley_journee/ -- Journee (1992) Wigley I forward-speed seakeeping data (secondary transcriptions)

Track B step B5 validation data (stage note `learn/meta/RANKINE_STAGE1_2026-10-06.md` section 10; registered in
`data/external/SOURCES.md`, section wigley_journee/). The original report (J.M.J. Journee, "Experiments and calculations
on four Wigley hullforms", TU Delft, Ship Hydromechanics Laboratory, report 909, 1992) could not be retrieved (the TU Delft
repository returns HTTP 403 to scripted access and lists the file under an indefinite embargo).

The CSV files are OUR transcriptions (numbers only) of the experimental markers that two open-access papers plot for
Journee's data, read from the PDF drawing commands of their vector figures by `studies/rankine_journee_data.py`:

- `journee1992_wigley1_fn020_*.csv` from J. Mar. Sci. Eng. 2021, 9, 87 (MDPI, CC BY 4.0), figures 12, 13, 14, 16;
- `journee1992_wigley1_fn030_*.csv` from F. Kara (2022), Ships and Offshore Structures, open-access version of record
  (Sheffield Hallam repository SHURA 29602), figure 9.

Columns: abscissa (omega' = omega sqrt(L/g), nu' = k L or lambda/L), value, reading uncertainty of the value and of the
abscissa (1/6 of the marker size plus the calibration residual). How accurately the papers' authors placed Journee's
values is not known (secondary transcription).

The two source PDFs are kept here for transcription traceability only (sha256 in SOURCES.md); they are excluded from any
published copy of the repository by `.gitignore` (licence terms of the second one not verified here).
