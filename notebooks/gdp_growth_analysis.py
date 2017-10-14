"""
GDP Growth Analysis
ECON 390 - Applied Econometrics
Fall 2017

looking at US GDP growth trends, recession indicators,
and whether yield curve inversion actually predicts anything
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from fredapi import Fred

FRED_KEY = os.environ.get('FRED_API_KEY')
fred = Fred(api_key=FRED_KEY)


def get_gdp_data(start='1970-01-01'):
    """pull real GDP and compute quarterly growth rates"""
    gdp = fred.get_series('GDPC1', observation_start=start)
    gdp = gdp.to_frame('real_gdp')
    gdp['growth_rate'] = gdp['real_gdp'].pct_change() * 100
    gdp['yoy_growth'] = gdp['real_gdp'].pct_change(4) * 100
    return gdp.dropna()


def get_recession_dates():
    """NBER recession indicator from FRED"""
    rec = fred.get_series('USREC', observation_start='1970-01-01')
    return rec


def get_yield_spread(start='1982-01-01'):
    """10yr minus 2yr treasury spread — classic recession predictor"""
    t10 = fred.get_series('DGS10', observation_start=start)
    t2 = fred.get_series('DGS2', observation_start=start)
    spread = (t10 - t2).dropna()
    spread.name = 'yield_spread'
    return spread


def plot_gdp_with_recessions(gdp, recessions):
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.bar(gdp.index, gdp['growth_rate'],
           width=80, color=['#d32f2f' if x < 0 else '#1976d2' for x in gdp['growth_rate']],
           alpha=0.7)

    # shade recession periods
    rec_starts = recessions[recessions.diff() == 1].index
    rec_ends = recessions[recessions.diff() == -1].index
    if len(rec_ends) < len(rec_starts):
        rec_ends = rec_ends.append(pd.DatetimeIndex([recessions.index[-1]]))

    for start, end in zip(rec_starts, rec_ends):
        ax.axvspan(start, end, alpha=0.15, color='gray')

    ax.set_title('US Real GDP Quarterly Growth Rate (1970-2017)')
    ax.set_ylabel('Growth Rate (%)')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    plt.tight_layout()
    plt.savefig('gdp_growth.png', dpi=150)
    plt.show()
    print('saved to gdp_growth.png')


def plot_yield_curve_vs_gdp(spread, gdp):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.plot(spread.index, spread.values, color='#1976d2', linewidth=0.8)
    ax1.axhline(y=0, color='red', linewidth=1, linestyle='--')
    ax1.fill_between(spread.index, spread.values, 0,
                     where=spread.values < 0, alpha=0.3, color='red')
    ax1.set_title('10Y-2Y Treasury Spread')
    ax1.set_ylabel('Spread (%)')

    ax2.plot(gdp.index, gdp['yoy_growth'], color='#388e3c', linewidth=1)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_title('Real GDP Year-over-Year Growth')
    ax2.set_ylabel('YoY Growth (%)')

    plt.tight_layout()
    plt.savefig('yield_vs_gdp.png', dpi=150)
    plt.show()
    print('saved to yield_vs_gdp.png')


def summary_stats(gdp):
    """basic summary table for the paper"""
    stats = {
        'mean_quarterly_growth': gdp['growth_rate'].mean(),
        'std_quarterly_growth': gdp['growth_rate'].std(),
        'median_quarterly_growth': gdp['growth_rate'].median(),
        'negative_quarters': (gdp['growth_rate'] < 0).sum(),
        'total_quarters': len(gdp),
        'pct_negative': (gdp['growth_rate'] < 0).mean() * 100,
        'max_growth': gdp['growth_rate'].max(),
        'min_growth': gdp['growth_rate'].min(),
    }
    for k, v in stats.items():
        print(f'{k}: {v:.2f}')
    return stats


if __name__ == '__main__':
    print('pulling GDP data...')
    gdp = get_gdp_data()
    recessions = get_recession_dates()

    print('\n--- GDP Summary Stats (1970-2017) ---')
    summary_stats(gdp)

    print('\nplotting...')
    plot_gdp_with_recessions(gdp, recessions)

    spread = get_yield_spread()
    plot_yield_curve_vs_gdp(spread, gdp)

    print('\ndone')
