# econ-data-notebooks

notebooks from econ 390 (applied econometrics). pulling data from FRED and doing basic analysis.

need a FRED API key to run these — get one at https://fred.stlouisfed.org/docs/api/api_key.html

## notebooks

- `gdp_growth_analysis.ipynb` — gdp growth trends, recessions, leading indicators
- `unemployment_dynamics.ipynb` — unemployment vs inflation (phillips curve stuff)
- `housing_market.ipynb` — case-shiller index analysis, mortgage rates correlation

## setup

```
pip install -r requirements.txt
```

set your FRED key:
```
export FRED_API_KEY=your_key_here
```
