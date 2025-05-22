# Src/icc_r2.py

import numpy as np
import matplotlib.pyplot as plt
import scienceplots

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


def plot_variance_components(model_dict):
    regions = []
    era_iccs = []
    country_iccs = []
    r2_marginals = []
    r2_conditionals = []
    r2_adjusted = []
    
    for region, results in model_dict.items():
        if results is None:
            continue
            
        regions.append(region)
        era_iccs.append(results['icc_era'] * 100)
        country_iccs.append(results['icc_country'] * 100)
        r2_marginals.append(results['r2_results']['R2_marginal'])
        r2_conditionals.append(results['r2_results']['R2_conditional'])
        r2_adjusted.append(results['r2_results']['R2_adjusted'])
    
    # Shorter region names for better display
    region_labels = [r.replace(' and ', ' \& ') for r in regions]

    y = np.arange(len(regions))
    height = 0.35
    colors = ['#97233F', '#2D4B8E', '#2C5F2D']
    
    def add_value_labels(ax, bars, r=False):
        for bar in bars:
            width = bar.get_width()
            if width > 0.1:  # Only show label if bar is visible
                if r:
                    ax.text(width, bar.get_y() + bar.get_height()/2.,
                        f'{width:.2f}',
                        ha='left', va='center', fontsize=8)
                    
                else:
                    ax.text(width, bar.get_y() + bar.get_height()/3.,
                            f'{width:.2f}\%',
                            ha='left', va='center', fontsize=8)
 

    def style_axis(ax, ncol=2):
        current_xlim = ax.get_xlim()
        max_value = current_xlim[1]
        
        extended_xlim = max_value * 1.10
        
        ax.set_xlim(0, extended_xlim)
        
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        leg = ax.legend(bbox_to_anchor=(0.5, -0.25),
                    loc='upper center',
                    ncol=ncol,
                    frameon=True,
                    borderpad=0.8,
                    handletextpad=0.5,
                    columnspacing=1.0)
        

    
    return y, height, colors, region_labels, era_iccs, country_iccs, r2_marginals, r2_conditionals, r2_adjusted, add_value_labels, style_axis




def plot_combined_metrics(events_dict, deaths_dict):
    regions = []
    events_era_iccs, events_country_iccs = [], []
    deaths_era_iccs, deaths_country_iccs = [], []
    events_r2_marg, events_r2_cond = [], []
    deaths_r2_marg, deaths_r2_cond = [], []
    
    for region in events_dict.keys():
        if events_dict[region] is not None and deaths_dict[region] is not None:
            regions.append(region)
            # Events data
            events_era_iccs.append(events_dict[region]['icc_era'] * 100)
            events_country_iccs.append(events_dict[region]['icc_country'] * 100)
            events_r2_marg.append(events_dict[region]['r2_results']['R2_marginal'])
            events_r2_cond.append(events_dict[region]['r2_results']['R2_conditional'])
            # Deaths data
            deaths_era_iccs.append(deaths_dict[region]['icc_era'] * 100)
            deaths_country_iccs.append(deaths_dict[region]['icc_country'] * 100)
            deaths_r2_marg.append(deaths_dict[region]['r2_results']['R2_marginal'])
            deaths_r2_cond.append(deaths_dict[region]['r2_results']['R2_conditional'])

    region_mapping = {
        'Middle East & North Africa': 'Middle East \n \& \n North Africa',
        'Sub-Saharan Africa': 'Sub-Saharan \n Africa',
        'Latin America & the Caribbean': 'Latin America \n \& \n Caribbean',
        'Asia & Pacific': 'Asia \& Pacific'
    }
    region_labels = [region_mapping.get(r.replace(' and ', ' & '), r) for r in regions]
    x = np.arange(len(regions))
    width = 0.1
    
    colors = {
        'era': '#97233F',
        'country': '#2D4B8E'
    }
    
    def add_value_labels(ax, bars, is_percentage=True):
        for bar in bars:
            height = bar.get_height()
            if height > 0.01:
                label = f'{height:.2f}{"\\%" if is_percentage else ""}'
                ax.text(bar.get_x() + bar.get_width()/2, height + (ax.get_ylim()[1] * 0.01),
                       label, ha='center', va='bottom', fontsize=8, rotation=90)

    # First plot - R² values
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    
    # Events R² (Marginal, then Conditional)
    bars1 = ax1.bar(x - 1.5*width, events_r2_marg, width,
                   label='Events: $R^2$ Marginal', color=colors['era'], alpha=0.7)
    bars2 = ax1.bar(x - 0.5*width, events_r2_cond, width,
                   label='Events: $R^2$ Conditional', color=colors['country'], alpha=0.7)
    
    # Deaths R² (Marginal, then Conditional)
    bars3 = ax1.bar(x + 0.5*width, deaths_r2_marg, width,
                   label='Deaths: $R^2$ Marginal', color=colors['era'], alpha=0.7, hatch='//')
    bars4 = ax1.bar(x + 1.5*width, deaths_r2_cond, width,
                   label='Deaths: $R^2$ Conditional', color=colors['country'], alpha=0.7, hatch='//')
    
    ax1.set_ylim(0, 1.2)
    ax1.set_yticks(np.arange(0, 1.1, 0.2))
    
    for bars in [bars1, bars2, bars3, bars4]:
        add_value_labels(ax1, bars, is_percentage=False)
    
    ax1.set_ylabel('Coefficient of Determination')
    ax1.set_xticks(x)
    ax1.set_xticklabels(region_labels)
    ax1.grid(False)
    
    plt.subplots_adjust(bottom=0.28)
    leg1 = ax1.legend(bbox_to_anchor=(0.5, -0.25),
                    loc='upper center',
                    ncol=2,
                    frameon=True,
                    borderpad=0.8,
                    handletextpad=0.5,
                    columnspacing=1.0)
    
    # Second plot - ICC values
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    
    bars5 = ax2.bar(x - 1.5*width, events_era_iccs, width,
                   label='Events Rate: Era ICC', color=colors['era'], alpha=0.7)
    bars6 = ax2.bar(x - 0.5*width, events_country_iccs, width,
                   label='Events Rate: Country ICC', color=colors['country'], alpha=0.7)
    
    bars7 = ax2.bar(x + 0.5*width, deaths_era_iccs, width,
                   label='Deaths Rate: Era ICC', color=colors['era'], alpha=0.7, hatch='//')
    bars8 = ax2.bar(x + 1.5*width, deaths_country_iccs, width,
                   label='Deaths Rate: Country ICC', color=colors['country'], alpha=0.7, hatch='//')
    
    max_icc = max(max(events_era_iccs), max(events_country_iccs), 
                 max(deaths_era_iccs), max(deaths_country_iccs))
    ax2.set_ylim(0, max_icc * 1.2)
    
    for bars in [bars5, bars6, bars7, bars8]:
        add_value_labels(ax2, bars)
    
    ax2.set_ylabel('Variance Explained (\%)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(region_labels)
    ax2.grid(False)
    
    plt.subplots_adjust(bottom=0.28)
    leg2 = ax2.legend(bbox_to_anchor=(0.5, -0.25),
                    loc='upper center',
                    ncol=2,
                    frameon=True,
                    borderpad=0.8,
                    handletextpad=0.5,
                    columnspacing=1.0)
    
    fig1.savefig('figs/r2_plot.pdf', bbox_inches='tight')
    fig2.savefig('figs/icc_plot.pdf', bbox_inches='tight')
    
    return fig1, fig2
