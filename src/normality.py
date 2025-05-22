# Src/normality.py

import numpy as np
import matplotlib.pyplot as plt

plt.style.use(['science'])
plt.rcParams["figure.figsize"] = (6, 4)
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.labelpad"] = 8
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["xtick.major.pad"] = 8
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["figure.autolayout"] = True
plt.rcParams["legend.fontsize"] = 12

plt.rcParams["legend.edgecolor"] = 'black'

def get_axis_limits(results_events, results_deaths, region_order):
    # Initialize with the first values
    x_min = float('inf')
    x_max = float('-inf')
    y_min = float('inf')
    y_max = float('-inf')
    
    # Find global min/max for both axes across all regions
    for region in region_order:
        # Get theoretical quantiles and ordered residuals for both models
        from scipy import stats
        
        # Events
        ev_resid = results_events[region]['model'].resid
        ev_theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, len(ev_resid)))
        
        # Deaths
        death_resid = results_deaths[region]['model'].resid
        death_theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, len(death_resid)))
        
        # Update limits
        x_min = min(x_min, ev_theoretical.min(), death_theoretical.min())
        x_max = max(x_max, ev_theoretical.max(), death_theoretical.max())
        y_min = min(y_min, ev_resid.min(), death_resid.min())
        y_max = max(y_max, ev_resid.max(), death_resid.max())
    
    # Add a small buffer (10%) to make sure points aren't right at the edges
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    return {
        'x': (x_min - 0.05 * x_range, x_max + 0.05 * x_range),
        'y': (y_min - 0.05 * y_range, y_max + 0.05 * y_range)
    }

def plot_single_region_qq(results_events, results_deaths, region, axis_limits):
    fig = plt.figure(figsize=(6, 4))
    
    from scipy import stats
    
    # Events QQ data
    ev_resid = results_events['model'].resid
    ev_theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, len(ev_resid)))
    ev_ordered = np.sort(ev_resid)
    
    # Deaths QQ data
    death_resid = results_deaths['model'].resid
    death_theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, len(death_resid)))
    death_ordered = np.sort(death_resid)
    
    # Remove outliers for Shapiro test
    def remove_outliers(x):
        Q1 = np.percentile(x, 25)
        Q3 = np.percentile(x, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return x[(x >= lower_bound) & (x <= upper_bound)]
    
    # Shapiro test on non-outlier data
    ev_clean = remove_outliers(ev_resid)
    death_clean = remove_outliers(death_resid)
    
    _, events_s_p = stats.shapiro(ev_clean)
    _, deaths_s_p = stats.shapiro(death_clean)
    
    # D'Agostino test
    _, events_d_p = stats.normaltest(ev_resid)
    _, deaths_d_p = stats.normaltest(death_resid)
    
    plt.scatter(ev_theoretical, ev_ordered, 
               alpha=0.65, color='#2D4B8E', s=50, label='Events Rate')
    plt.scatter(death_theoretical, death_ordered,
               alpha=0.65, color='#97233F', s=50, label='Death Rate')
    
    # Cleveland reference lines using numpy's percentile function
    def extend_line(p25, p75, y25, y75, xlims):
        # Calculate slope and intercept from quartile points
        slope = (y75 - y25) / (p75 - p25)
        intercept = y25 - slope * p25

        x = np.array(xlims)
        y = slope * x + intercept
        return x, y
    
    # Events Rate
    ev_p25 = stats.norm.ppf(0.25)  # theoretical 25th percentile
    ev_p75 = stats.norm.ppf(0.75)  # theoretical 75th percentile
    ev_y25 = np.percentile(ev_ordered, 25)  # sample 25th percentile
    ev_y75 = np.percentile(ev_ordered, 75)  # sample 75th percentile
    
    x_ev, y_ev = extend_line(ev_p25, ev_p75, ev_y25, ev_y75, axis_limits['x'])
    plt.plot(x_ev, y_ev, '-', color='#2D4B8E', alpha=0.8, linewidth=1.5, zorder=1)
    
    # Deaths Rate
    death_p25 = stats.norm.ppf(0.25)  # theoretical 25th percentile
    death_p75 = stats.norm.ppf(0.75)  # theoretical 75th percentile
    death_y25 = np.percentile(death_ordered, 25)  # sample 25th percentile
    death_y75 = np.percentile(death_ordered, 75)  # sample 75th percentile
    
    x_death, y_death = extend_line(death_p25, death_p75, death_y25, death_y75, axis_limits['x'])
    plt.plot(x_death, y_death, '-', color='#97233F', alpha=0.8, linewidth=1.5, zorder=1)
    
    plt.xlim(axis_limits['x'])
    plt.ylim(axis_limits['y'])
    
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    
    # Status indicators with both tests
    def get_status(s_p, d_p):
        s_status = "Normal" if s_p > 0.05 else "Non-Normal"
        d_status = "Normal" if d_p > 0.05 else "Non-Normal"
        return f"{s_status}$_{{\mathrm{{Shapiro}}}}  (p = {s_p:.3f})$ $|$ {d_status}$_{{\mathrm{{D'Agostino}}}}  (p = {d_p:.3f})$"
    
    events_status = get_status(events_s_p, events_d_p)
    deaths_status = get_status(deaths_s_p, deaths_d_p)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='#2D4B8E', linestyle='-',
               label=f'Events Rate: {events_status}', 
               markersize=8, markeredgewidth=0),
        Line2D([0], [0], marker='o', color='#97233F', linestyle='-',
               label=f'Death Rate: {deaths_status}', 
               markersize=8, markeredgewidth=0)
    ]

    leg = plt.legend(handles=legend_elements,
                    bbox_to_anchor=(0.5, -0.25),
                    loc='upper center',
                    ncol=1,
                    frameon=True,
                    borderpad=0.8,
                    handletextpad=0.5,
                    columnspacing=1.0)

    plt.tight_layout()
    return fig
