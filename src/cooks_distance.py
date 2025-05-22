# Src/cooks_distance.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import statsmodels.api as sm


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

def calculate_cooks_distance(model_result, X, y):
    y_pred = model_result.predict()
    residuals = y - y_pred
    
    # Calculate leverage (hat matrix diagonal)
    leverage = np.zeros(len(X))

    # Get fixed effects design matrix
    X_fixed = X.values
    
    # Calculate approximate hat matrix diagonal
    V = model_result.cov_params()
    H = X_fixed @ V @ X_fixed.T
    leverage = np.diag(H)

    
    # Calculate MSE
    mse = np.sum(residuals**2) / (len(y) - len(model_result.fe_params))
    
    # Calculate Cook's distance
    cooks_d = (residuals**2 / (len(model_result.fe_params) * mse)) * \
              (leverage / (1 - leverage)**2)
    
    return pd.Series(cooks_d, index=y.index)

def analyze_cooks_distance(model_result, data, X, y, threshold=None):
    cooks_d = calculate_cooks_distance(model_result, X, y)
    
    # Set threshold
    if threshold is None:
        threshold = 4/len(cooks_d)
    
    outliers = cooks_d > threshold
    
    results_df = pd.DataFrame({
        'cooks_distance': cooks_d,
        'is_outlier': outliers,
        'country': data['country_id'],
        'era': data['era']
    })
    
    return {
        'cooks_distances': cooks_d,
        'outliers': outliers,
        'threshold': threshold,
        'results_df': results_df
    }


def plot_cooks_distance(analysis_results, data, region_name=None, figsize=(6, 4)):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from collections import defaultdict
    import statsmodels.api as sm
    
    fig, ax = plt.subplots(figsize=figsize)
    
    cooks_d = analysis_results['cooks_distances']
    results_df = analysis_results['results_df']
    threshold = analysis_results['threshold']
    
    # Set y-axis limits with 15% extra space
    y_max = max(cooks_d) * 1.15
    ax.set_ylim(-0.01, y_max)
    
    markerline, stemlines, baseline = ax.stem(
        range(len(cooks_d)), 
        cooks_d,
        markerfmt='.',
        linefmt='-',
        basefmt='gray',
    )
    markerline.set_color('#2D4B8E')
    plt.setp(markerline, markersize=4, color='#2D4B8E')

    plt.setp(stemlines, linewidth=1, alpha=0.7, zorder=1, color ='#2D4B8E')  # Lower zorder for stems
    
    # Add threshold line
    ax.axhline(y=threshold, color='#97233F', linestyle='--', linewidth=1.5, alpha=0.8, zorder=2)
    
    # Get outliers and group by country
    outlier_df = results_df[results_df['is_outlier']].copy()
    country_map = dict(zip(data['country_id'], data['country_name']))
    outlier_df['country_name'] = outlier_df['country'].map(country_map)
    
    # Define LaTeX symbols for countries
    latex_symbols = {
        country: symbol for country, symbol in zip(
            sorted(outlier_df['country_name'].unique()),
            [ r'$\Delta$', r'$\Xi$', r'$\varkappa$', r'$\bigcap$', r'$ta$', r'$mu$',
             r'$\zeta$', r'$\eta$', r'$\theta$', r'$\iota$', r'$\kappa$',
             r'$\lambda$', r'$\mu$', r'$\nu$', r'$\xi$', r'$\pi$']
        )
    }
    
    country_groups = defaultdict(list)
    country_details = defaultdict(list)
    for idx, row in outlier_df.iterrows():
        country_groups[row['country_name']].append((idx, row['cooks_distance']))
        country_details[row['country_name']].append(f"Era: {row['era']} ({row['cooks_distance']:.3f})")

    legend_elements = [
        mpatches.Patch(color='#97233F', linestyle='--', alpha=0.8,
                      label=f"Cook's distance Threshold ({threshold:.3f})")
    ]
    
    ax.legend(handles=legend_elements,
             bbox_to_anchor=(0.5, -0.25),
             loc='upper center',
             ncol=1,
             frameon=True,
             fontsize=10)
    
    y_max = max(cooks_d)
    y_min = min(cooks_d)
    y_range = y_max - y_min
    
    used_positions = []
    
    # Add symbols for outliers
    for idx in outlier_df.index:
        x = idx
        y = outlier_df.loc[idx, 'cooks_distance']
        country_name = outlier_df.loc[idx, 'country_name']
        symbol = latex_symbols[country_name]
        
        # Calculate initial position with smaller offset
        y_text = y + y_range * 0.02
        
        # For points near the top of the plot, place symbol below instead
        if y > y_max * 0.8:
            y_text = y - y_range * 0.02
        
        # Check for overlaps
        attempts = 0
        base_offset = y_range * 0.02
        while attempts < 10 and any(abs(pos[0] - x) < 3 and abs(pos[1] - y_text) < y_range * 0.03 for pos in used_positions):
            if attempts % 2 == 0:
                # Try alternating left/right offset
                x_offset = (attempts + 1) * 0.5
                x = x + (-1)**(attempts//2) * x_offset
            else:
                # Try different vertical positions
                y_text = y + (-1)**(attempts//2) * base_offset * (attempts + 1)
            attempts += 1
        
        used_positions.append((x, y_text))
        
        ax.text(x, y_text, symbol, 
                ha='center',
                va='bottom',
                fontsize=10,
                color='black',
                zorder=3,
                bbox=dict(facecolor='white', 
                         edgecolor='none',
                         alpha=0.5,
                         pad=0.1))
    
    # Add country details as text box with symbols
    detail_text = []
    for country, details in sorted(country_details.items()):
        # Add country name with its symbol
        detail_text.append(f"{country} ({latex_symbols[country]})")
        for detail in details:
            detail_text.append(f"  {detail}")
        # Add spacing between countries
        detail_text.append("")
    
    if detail_text:
        if detail_text[-1] == "":
            detail_text.pop()
            
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        y_pos = min(0.98, 0.98 - 0.02 * len(country_details))
        ax.text(1.02, y_pos, '\n'.join(detail_text),
                transform=ax.transAxes,
                verticalalignment='top',
                fontsize=10,
                linespacing=1.3,
                bbox=props)
    
    ax.set_xlabel('Observation', fontsize=14)
    ax.set_ylabel("Cook's Distance", fontsize=14)
    
    ax.grid(False)
    
    plt.subplots_adjust(right=0.85)
    
    return fig

def analyze_region_outliers(region_results, data, predictors):
    if region_results is None:
        print(f"No results available for analysis")
        return None
        
    try:
        X = sm.add_constant(data[predictors])
        y = data[region_results['prediction_column']]

        analysis_results = analyze_cooks_distance(
            region_results['model'],
            data,
            X,
            y
        )

        fig = plot_cooks_distance(
            analysis_results,
            data,
            region_name=region_results['region'],
            figsize=(6, 4) 
            
        )
        
        return {
            'analysis': analysis_results,
            'plot': fig
        }
        
    except Exception as e:
        print(f"Error in outlier analysis: {str(e)}")
        return None
    



"""
DRIVER CODE

def run_outlier_analysis(results_dict, df, predictors, prediction_column):
    for region, results in results_dict.items():
        print(f"\nAnalyzing outliers for {region}")
        print("-" * (20 + len(region)))
        
        results['prediction_column'] = prediction_column
        results['region'] = region
        
        region_data = prepare_data(df, prediction_column, predictors, region)
        
        # Run outlier analysis
        outlier_analysis = analyze_region_outliers(results, region_data, predictors)
        
        if outlier_analysis:
            plt.figure(outlier_analysis['plot'].number)
            plt.savefig(f'figs/{region}_events_rate_outliers.pdf', bbox_inches='tight')
            plt.show()
            
            # Print outlier details
            outliers = outlier_analysis['analysis']['results_df'][
                outlier_analysis['analysis']['results_df']['is_outlier']
            ]
            if len(outliers) > 0:
                print(f"\nOutliers in {region}:")
                print(outliers[['country', 'era', 'cooks_distance']].sort_values('cooks_distance', ascending=False))
            else:
                print(f"\nNo outliers detected in {region}")
                
            plt.close() 

print("Analyzing outliers for events rate")
run_outlier_analysis(results_events, df_filtered, predictors, 'events_rate')


"""
