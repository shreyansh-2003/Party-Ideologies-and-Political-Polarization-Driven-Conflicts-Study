# Src/coeff_importance.py

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
matplotlib.interactive(False)

import scienceplots

plt.style.use(['science'])
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

def extract_fixed_effects(model_result):
    param_names = model_result.model.exog_names
    
    # Extract the fixed effects parameters
    fe_params = model_result.fe_params
    cov_matrix = model_result.cov_params()
    
    fe_summary = pd.DataFrame({
        'coef': fe_params,
        'std_err': np.sqrt(np.diag(cov_matrix))[:len(param_names)]
    }, index=param_names)
    
    # Calculate z-scores and p-values
    fe_summary['z_value'] = fe_summary['coef'] / fe_summary['std_err']
    fe_summary['p_value'] = 2 * (1 - stats.norm.cdf(abs(fe_summary['z_value'])))
    
    # Calculate confidence intervals
    fe_summary['ci_lower'] = fe_summary['coef'] - 1.96 * fe_summary['std_err']
    fe_summary['ci_upper'] = fe_summary['coef'] + 1.96 * fe_summary['std_err']
    
    # Mark significant coefficients
    fe_summary['significant'] = fe_summary['p_value'] < 0.05
    
    return fe_summary

def plot_region_indv_coefficients(model_result, region, figsize=(10, 6)):
    try:
        fe_summary = extract_fixed_effects(model_result)
        
        if 'const' in fe_summary.index:
            fe_summary = fe_summary.drop('const')
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = ['#2C5F2D','#97233F']  # muted green and red
        
        y_pos = np.arange(len(fe_summary))
        
        # Error bars
        ax.hlines(y_pos, fe_summary['ci_lower'], fe_summary['ci_upper'],
                 color='gray', alpha=0.3, linewidth=1.5, zorder=1)
        
        # Plot coefficients with larger, more visible points
        ax.scatter(fe_summary['coef'], y_pos,
                  c=[colors[0] if sig else colors[1] for sig in fe_summary['significant']],
                  s=70, zorder=2)
        
        # Add vertical line at 0
        ax.axvline(x=0, color='#2D4B8E', linestyle='--', alpha=0.7, zorder=0)
        

        # Another way to do it:
        ylabels = [label if not sig else f"$\\bf{{{label}}}$" 
                for label, sig in zip(fe_summary.index, fe_summary['significant'])]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(ylabels)

        ax.set_xlabel('Coefficient Value')
        
        ax.grid(True, axis='x', alpha=0.2, linestyle='-')
        
        plt.subplots_adjust(left=0.45, right=0.95, bottom=0.15, top=0.95)

        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=colors[0], label='Significant (p $<$ 0.05)',
                      markersize=8, markeredgewidth=0),
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=colors[1], label='Non-significant (p $>$ 0.05)',
                      markersize=8, markeredgewidth=0)
        ]
        
        leg = ax.legend(handles=legend_elements,
                       bbox_to_anchor=(0.5, -0.125),
                       loc='upper center',
                       ncol=2,
                       frameon=True,
                       borderpad=0.8,
                       handletextpad=0.5,
                       columnspacing=1.0)
        
                        
        plt.tight_layout()
        
        return fig, fe_summary
        
    except Exception as e:
        print(f"Error plotting region {region}: {str(e)}")
        return None, None
    


def plot_region_coefficients(model_results_dict, control_variables, figsize=(10, 6)):
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()
    
    all_coeffs = []
    for region_data in model_results_dict.values():
        model = region_data['model']
        param_names = model.model.exog_names
        coef = model.fe_params
        stderr = np.sqrt(np.diag(model.cov_params()))[:len(param_names)]
        pvalues = 2 * (1 - stats.norm.cdf(abs(coef / stderr)))
        sig_coeffs = [name for name, p in zip(param_names, pvalues) if p < 0.05]
        all_coeffs.extend(sig_coeffs)
    
    coeff_counts = {x: all_coeffs.count(x) for x in set(all_coeffs)}
    unique_coeffs = {coeff for coeff, count in coeff_counts.items() if count == 1}
    
    y_positions = []
    coeff_labels = []
    current_y = 0
    region_positions = []
    region_labels = []
    separator_positions = []
    
    # Process each region
    for region_name, region_data in model_results_dict.items():
        model = region_data['model']
        
        # Get parameter info
        param_names = model.model.exog_names
        coef = model.fe_params
        stderr = np.sqrt(np.diag(model.cov_params()))[:len(param_names)]
        pvalues = 2 * (1 - stats.norm.cdf(abs(coef / stderr)))
        
        sig_coeffs = pd.DataFrame({
            'coef': coef,
            'stderr': stderr,
            'pvalue': pvalues,
            'ci_lower': coef - 1.96 * stderr,
            'ci_upper': coef + 1.96 * stderr,
            'name': param_names
        }, index=param_names)
        
        # Filter for significant coefficients
        sig_coeffs = sig_coeffs[sig_coeffs['pvalue'] < 0.05]

        # sort coeff in ascending order by name
        sig_coeffs = sig_coeffs.sort_index()
        
        if not sig_coeffs.empty:
            y_pos = np.arange(len(sig_coeffs)) + current_y
            
            ax1.hlines(y_pos, sig_coeffs['ci_lower'], sig_coeffs['ci_upper'],
                        color='gray', alpha=0.6, linewidth=1.5)
            
            for idx, (name, row) in enumerate(sig_coeffs.iterrows()):
                if name in control_variables:
                    color = '#2C5F2D'  # green for control
                    marker = 's'
                else:
                    color = '#2C5F2D'  # green for polarization
                    marker = '^'
                
                ax1.scatter(row['coef'], y_pos[idx], 
                            marker=marker, s=100, 
                            color=color)
                
                # Add red line through center if unique
                if name in unique_coeffs:
                    if marker == 's': 
                        line_length = 0.1  
                        ax1.plot([row['coef'] - line_length/2, row['coef'] + line_length/2],
                                [y_pos[idx], y_pos[idx]],
                                color='#97233F', linewidth=2, zorder=3)
                    else:  # For triangles
                        line_length = 0.1  
                        ax1.plot([row['coef'] - line_length/2, row['coef'] + line_length/2],
                                [y_pos[idx], y_pos[idx]],
                                color='#97233F', linewidth=2, zorder=3)
            
            y_positions.extend(y_pos)
            

            # include space between coefficients words
            coeff_labels.extend([f"$\\textbf{{{name}}}$" if name in unique_coeffs else name
                                for name in sig_coeffs.index])

                            
            region_positions.append(np.mean(y_pos))
            region_labels.append(region_name)
            
            separator_positions.append(max(y_pos) + 1)
            current_y = max(y_pos) + 2
    
    # Add region separators
    xlims = ax1.get_xlim()
    for sep_pos in separator_positions[:-1]:
        ax1.axhline(y=sep_pos, color='gray', linestyle='-', alpha=0.25, linewidth=1)
        ax1.fill_between([xlims[0], xlims[1]], sep_pos-0.45, sep_pos+0.45, 
                        color='gray', alpha=0.075)
    
    # Add vertical line at 0
    ax1.axvline(x=0, color='#2D4B8E', linestyle='--', alpha=0.7)

    region_mapping = {
        'Middle East & North Africa': 'Middle East \n \& \n North Africa',
        'Sub-Saharan Africa': 'Sub-Saharan \n Africa',
        'Latin America & the Caribbean': 'Latin America \n \& \n Caribbean',
        'Asia & Pacific': 'Asia \& Pacific'
    }
    
    mapped_region_labels = [region_mapping.get(r.replace(' and ', ' & '), r) for r in region_labels]


    ax1.set_yticks(region_positions)
    ax1.set_yticklabels(mapped_region_labels, ha='right')
    
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(coeff_labels, ha='left')
    
    ax1.set_xlabel('Coefficient Value')
    ax1.grid(False)


    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#2c5f2d',
                    label='Control Variable', markersize=10),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#2c5f2d',
                    label='Polarization Variable', markersize=10),
        plt.Line2D([0], [0], marker='_', color='#97233F', markerfacecolor='#2c5f2d',
                    label='Unique to One Region', markersize=10)]

    
    
    leg = plt.legend(handles=legend_elements,
                    bbox_to_anchor=(0.5, -0.15),
                    loc='upper center',
                    ncol=3,
                    frameon=True,
                    borderpad=0.8,
                    handletextpad=0.5,
                    columnspacing=1.0)
    
    # Adjust layout
    plt.subplots_adjust(left=0.25, right=0.75, bottom=0.2)
    
    return fig


"""
DRIVER CODE
# Individual coeffs

for region, model_results in results_events.items():
    if model_results is None or 'model' not in model_results:
        continue
        
    fig, fe_summary = plot_region_indv_coefficients(model_results['model'], 
                                                region)


# All coeffs

fig = plot_region_coefficients(results_events, control_variables, polarization_columns)
plt.show()

"""