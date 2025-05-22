# Src/hetroskedasticity.py

import matplotlib.pyplot as plt
import scienceplots
import numpy as np
from scipy.stats import pearsonr

plt.style.use(['science'])
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.labelpad"] = 8
plt.rcParams["xtick.labelsize"] = 15
plt.rcParams["xtick.major.pad"] = 8
plt.rcParams["ytick.labelsize"] = 15
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["figure.autolayout"] = True
plt.rcParams["legend.fontsize"] = 16

plt.rcParams["legend.edgecolor"] = 'black'


def plot_single_region_residuals(results_events, results_deaths):
    fig = plt.figure(figsize=(9, 5))
    ev_fitted = results_events['model'].fittedvalues
    death_fitted = results_deaths['model'].fittedvalues
    ev_resid = results_events['model'].resid
    death_resid = results_deaths['model'].resid
    
    # Standardize all values
    ev_fitted_std = (ev_fitted - np.mean(ev_fitted)) / np.std(ev_fitted)
    death_fitted_std = (death_fitted - np.mean(death_fitted)) / np.std(death_fitted)
    ev_resid_std = (ev_resid - np.mean(ev_resid)) / np.std(ev_resid)
    death_resid_std = (death_resid - np.mean(death_resid)) / np.std(death_resid)
    
    plt.scatter(ev_fitted_std, ev_resid_std,
               alpha=0.65, color='#2D4B8E', s=50, label='Events Rate')
    plt.scatter(death_fitted_std, death_resid_std,
               alpha=0.65, color='#97233F', s=50, label='Death Rate')
    
    # Zero line
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Calculate correlations for linearity test (using absolute residuals)
    ev_corr, ev_p = pearsonr(ev_fitted_std, np.abs(ev_resid_std))
    death_corr, death_p = pearsonr(death_fitted_std, np.abs(death_resid_std))
    
    plt.xlabel("Standardized Fitted Values", fontsize=16)
    plt.ylabel("Standardized Residuals", fontsize=16)
    
    # Get heteroskedasticity status with combined tests
    def get_status(bp_p, corr, corr_p):
        hetero_status = "Homoscedastic" if bp_p > 0.05 else "Heteroskedastic"

        # Determine strength of correlation, <0.2 very weak, 0.2-0.4 weak, 0.4-0.6 moderate, 0.6-0.8 strong, >0.8 very strong
        corr_strength = "Very-Weak" if abs(corr) < 0.2 else "Weak" if abs(corr) < 0.4 else "Moderate" if abs(corr) < 0.6 else "Strong" if abs(corr) < 0.8 else "Very-Strong"
        pattern_status = f"{corr_strength}-Pattern" if corr_p <= 0.05 else "No-Pattern"
        
        return f"{hetero_status}$_{{\mathrm{{BP}}}}(p = {bp_p:.3f})$ $|$ {pattern_status}$_{{\mathrm{{corr}}}}(r = {corr:.3f}, p = {corr_p:.3f})$"
    
    events_status = get_status(
        results_events['heteroskedasticity']['bp_pvalue'],
        ev_corr,
        ev_p
    )
    
    deaths_status = get_status(
        results_deaths['heteroskedasticity']['bp_pvalue'],
        death_corr,
        death_p
    )
    
    events_label = f'Events Rate: {events_status}'
    deaths_label = f'Death Rate: {deaths_status}'
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2D4B8E', 
               label=events_label, markersize=8, markeredgewidth=0),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#97233F', 
               label=deaths_label, markersize=8, markeredgewidth=0)
    ]
    
    leg = plt.legend(handles=legend_elements,
                    bbox_to_anchor=(0.5, -0.30),
                    loc='upper center',
                    ncol=1,
                    frameon=True,
                    borderpad=0.8,
                    handletextpad=0.5,
                    columnspacing=1.0)
    
    plt.tight_layout()
    return fig

"""
DRIVER CODE
Heterogenity Plot with P Value Scoes


from visualisations.hetroskedasticity_lr import get_axis_limits, plot_single_region_residuals

# Driver CODE

region_order = [
    'Asia and Pacific',
    'Latin America and the Caribbean', 
    'Middle East and North Africa',
    'Sub-Saharan Africa'
]

for region in region_order:
    fig = plot_single_region_residuals(
        results_events[region],
        results_deaths[region],
    )
    plt.show()

"""