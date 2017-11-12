"""
Housing Market Analysis
ECON 390 - Applied Econometrics
Fall 2017

case-shiller home price index vs mortgage rates, construction starts,
and income growth. wanted to see if the 2008 bubble is obvious in hindsight
with basic metrics. (it is)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fredapi import Fred

FRED_KEY = os.environ.get('FRED_API_KEY')
fred = Fred(api_key=FRED_KEY)


def get_housing_data():
    """pull housing-related series from FRED"""
    series = {
        'case_shiller': 'CSUSHPINSA',     # Case-Shiller Home Price Index
        'mortgage_30yr': 'MORTGAGE30US',   # 30-year fixed mortgage rate
        'housing_starts': 'HOUST',         # housing starts (thousands)
        'median_income': 'MEHOINUSA672N',  # median household income (annual)
        'home_ownership': 'RHORUSQ156N',   # homeownership rate
    }

    data = {}
    for name, code in series.items():
        try:
            s = fred.get_series(code, observation_start='1990-01-01')
            s.name = name
            data[name] = s
            print(f'  got {name}: {len(s)} observations')
        except Exception as e:
            print(f'  failed to get {name}: {e}')

    return data


def price_to_income_ratio(case_shiller, median_income):
    """
    normalize case-shiller to a price-to-income ratio
    this is crude but shows the bubble clearly
    """
    # resample case-shiller to annual to match income
    cs_annual = case_shiller.resample('YE').mean()
    merged = pd.concat([cs_annual, median_income], axis=1).dropna()
    merged.columns = ['home_price_idx', 'median_income']

    # normalize both to 100 at start
    merged['price_norm'] = merged['home_price_idx'] / merged['home_price_idx'].iloc[0] * 100
    merged['income_norm'] = merged['median_income'] / merged['median_income'].iloc[0] * 100
    merged['ratio'] = merged['price_norm'] / merged['income_norm']
    return merged


def plot_bubble_indicators(data):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # case-shiller
    cs = data['case_shiller']
    axes[0].plot(cs.index, cs.values, color='#d32f2f', linewidth=1.5)
    axes[0].set_title('Case-Shiller Home Price Index')
    axes[0].set_ylabel('Index')
    axes[0].axvspan('2007-01-01', '2012-01-01', alpha=0.1, color='red')

    # mortgage rates
    mr = data['mortgage_30yr']
    axes[1].plot(mr.index, mr.values, color='#1976d2', linewidth=1)
    axes[1].set_title('30-Year Fixed Mortgage Rate')
    axes[1].set_ylabel('Rate (%)')
    axes[1].axhline(y=mr.mean(), color='gray', linestyle='--', alpha=0.5, label=f'avg: {mr.mean():.1f}%')
    axes[1].legend()

    # housing starts
    hs = data['housing_starts']
    axes[2].plot(hs.index, hs.values, color='#388e3c', linewidth=1)
    axes[2].set_title('Housing Starts (Thousands of Units)')
    axes[2].set_ylabel('Starts (000s)')
    axes[2].axhline(y=hs.mean(), color='gray', linestyle='--', alpha=0.5)

    plt.suptitle('Housing Market Indicators (1990-2017)', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig('housing_indicators.png', dpi=150)
    plt.show()


def plot_price_income(ratio_df):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(ratio_df.index, ratio_df['price_norm'], label='Home Prices (normalized)', color='#d32f2f', linewidth=2)
    ax.plot(ratio_df.index, ratio_df['income_norm'], label='Median Income (normalized)', color='#1976d2', linewidth=2)

    ax.fill_between(ratio_df.index,
                    ratio_df['price_norm'], ratio_df['income_norm'],
                    where=ratio_df['price_norm'] > ratio_df['income_norm'],
                    alpha=0.2, color='red', label='Price > Income growth')

    ax.set_title('Home Prices vs Income Growth (Normalized to 100)')
    ax.set_ylabel('Index (1990 = 100)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('price_vs_income.png', dpi=150)
    plt.show()


def compute_appreciation_rates(case_shiller):
    """annual and 5-year rolling appreciation"""
    annual = case_shiller.resample('YE').mean()
    annual_pct = annual.pct_change() * 100
    rolling_5yr = annual.pct_change(5) * 100

    print('\n--- Annual Appreciation Stats ---')
    print(f'  mean: {annual_pct.mean():.1f}%')
    print(f'  median: {annual_pct.median():.1f}%')
    print(f'  max: {annual_pct.max():.1f}% ({annual_pct.idxmax().year})')
    print(f'  min: {annual_pct.min():.1f}% ({annual_pct.idxmin().year})')

    return annual_pct


if __name__ == '__main__':
    print('pulling housing data from FRED...')
    data = get_housing_data()

    print('\nplotting indicators...')
    plot_bubble_indicators(data)

    if 'case_shiller' in data and 'median_income' in data:
        ratio_df = price_to_income_ratio(data['case_shiller'], data['median_income'])
        plot_price_income(ratio_df)

    if 'case_shiller' in data:
        compute_appreciation_rates(data['case_shiller'])

    print('\n--- observations ---')
    print('1. home prices completely decoupled from income growth 2002-2006')
    print('2. housing starts peaked in 2005 — a full year before prices peaked')
    print('3. in hindsight the bubble is painfully obvious in the data')
    print('4. current (2017) price-to-income ratio is approaching 2006 levels again...')
