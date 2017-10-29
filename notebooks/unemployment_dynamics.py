"""
Unemployment Dynamics & Phillips Curve
ECON 390 - Applied Econometrics
Fall 2017

testing whether the phillips curve relationship still holds
in recent data. spoiler: it's complicated
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fredapi import Fred
from statsmodels.api import OLS, add_constant

FRED_KEY = os.environ.get('FRED_API_KEY')
fred = Fred(api_key=FRED_KEY)


def get_unemployment(start='1960-01-01'):
    u = fred.get_series('UNRATE', observation_start=start)
    u.name = 'unemployment_rate'
    return u


def get_inflation(start='1960-01-01'):
    """CPI-based inflation, year over year"""
    cpi = fred.get_series('CPIAUCSL', observation_start=start)
    inflation = cpi.pct_change(12) * 100  # yoy
    inflation.name = 'inflation_rate'
    return inflation.dropna()


def get_labor_force_participation(start='1960-01-01'):
    lfpr = fred.get_series('CIVPART', observation_start=start)
    lfpr.name = 'lfpr'
    return lfpr


def phillips_curve_analysis(unemployment, inflation):
    """
    run OLS regression: inflation = a + b * unemployment + e
    do it for different time periods to show structural break
    """
    merged = pd.concat([unemployment, inflation], axis=1).dropna()
    merged.columns = ['unemployment', 'inflation']

    periods = {
        'full_sample': ('1960', '2017'),
        'pre_volcker': ('1960', '1979'),
        'post_volcker': ('1983', '2000'),
        'post_2008': ('2009', '2017'),
    }

    results = {}
    for name, (start, end) in periods.items():
        subset = merged.loc[start:end]
        X = add_constant(subset['unemployment'])
        y = subset['inflation']
        model = OLS(y, X).fit()
        results[name] = {
            'coef': model.params['unemployment'],
            'pvalue': model.pvalues['unemployment'],
            'r_squared': model.rsquared,
            'n_obs': int(model.nobs),
        }
        print(f'\n--- {name} ({start}-{end}) ---')
        print(f'  beta: {model.params["unemployment"]:.3f} (p={model.pvalues["unemployment"]:.4f})')
        print(f'  R²: {model.rsquared:.3f}')
        print(f'  n: {int(model.nobs)}')

    return results, merged


def plot_phillips_scatter(merged):
    """scatter plot with color by decade"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    periods = [
        ('1960', '1979', 'Pre-Volcker (1960-1979)'),
        ('1983', '2000', 'Post-Volcker (1983-2000)'),
        ('2001', '2008', 'Pre-Crisis (2001-2008)'),
        ('2009', '2017', 'Post-Crisis (2009-2017)'),
    ]
    colors = ['#1976d2', '#d32f2f', '#388e3c', '#f57c00']

    for ax, (start, end, title), color in zip(axes.flat, periods, colors):
        subset = merged.loc[start:end]
        ax.scatter(subset['unemployment'], subset['inflation'],
                   alpha=0.3, s=10, color=color)

        # fit line
        z = np.polyfit(subset['unemployment'], subset['inflation'], 1)
        p = np.poly1d(z)
        x_range = np.linspace(subset['unemployment'].min(), subset['unemployment'].max(), 100)
        ax.plot(x_range, p(x_range), color=color, linewidth=2)

        ax.set_title(title)
        ax.set_xlabel('Unemployment Rate (%)')
        ax.set_ylabel('Inflation Rate (%)')

    plt.suptitle('Phillips Curve Across Different Eras', fontsize=14)
    plt.tight_layout()
    plt.savefig('phillips_curve.png', dpi=150)
    plt.show()


def plot_unemployment_timeline(unemployment, lfpr):
    fig, ax1 = plt.subplots(figsize=(14, 6))

    ax1.plot(unemployment.index, unemployment.values, color='#d32f2f', linewidth=1, label='Unemployment Rate')
    ax1.set_ylabel('Unemployment Rate (%)', color='#d32f2f')
    ax1.set_xlabel('Year')

    ax2 = ax1.twinx()
    ax2.plot(lfpr.index, lfpr.values, color='#1976d2', linewidth=1, alpha=0.7, label='LFPR')
    ax2.set_ylabel('Labor Force Participation Rate (%)', color='#1976d2')

    ax1.set_title('Unemployment Rate vs Labor Force Participation (1960-2017)')
    plt.tight_layout()
    plt.savefig('unemployment_lfpr.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    print('pulling data from FRED...')
    unemployment = get_unemployment()
    inflation = get_inflation()
    lfpr = get_labor_force_participation()

    print('\n=== Phillips Curve Regression Results ===')
    results, merged = phillips_curve_analysis(unemployment, inflation)

    print('\nplotting...')
    plot_phillips_scatter(merged)
    plot_unemployment_timeline(unemployment, lfpr)

    # takeaway
    print('\n--- takeaway ---')
    print('phillips curve relationship is way weaker post-2008.')
    print('the slope flattened significantly, inflation barely responds')
    print('to unemployment changes anymore. anchored expectations?')
