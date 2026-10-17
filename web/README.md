# Web showcase

A static, scroll-driven site that presents the project's story and results.
Plain HTML, CSS and JavaScript — no build step, no dependencies (fonts load from
Google Fonts, with system fallbacks offline).

## Open it

Serve the **repository root** (the page loads EDA figures from `../results/figures/`):

```bash
python -m http.server 8000
```

Then open http://localhost:8000/web/. Double-clicking `web/index.html` also works.

## Files

| File | Purpose |
|---|---|
| `index.html` | All chapters and copy |
| `css/style.css` | Theme tokens (light + dark), layout, components |
| `js/data.js` | Every number shown on the page, from `reports/final_report.md` |
| `js/charts.js` | Hand-built SVG charts: grouped bars, DP line chart, feature bars, rule tree |
| `js/main.js` | Scroll effects, federation stepper, ε slider, secure-aggregation demo, risk explorer |

## Updating numbers

Re-run the experiments, then edit `js/data.js`. Charts and text read from it.

## Deploying to GitHub Pages

Enable Pages on the repository (branch root) and visit `/web/`. Keep `results/figures/`
committed so the figures resolve.
