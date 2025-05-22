# Party Ideologies and Political Polarization-Driven Conflicts: A Study of the Global South

This repository contains the code and data for the paper "Party Ideologies and Political Polarization-Driven Conflicts: A Study of the Global South".
![Global Conflict Intensity Map](assets/figs/enhanced_conflict_map.pdf)


## Abstract
This study examines how political identity-induced polarization affects armed conflicts in Global South regions. Using polarization indices based on party stances from the V-Party Dataset and conflict data from UCDP, we analyze the relationship between political fragmentation and violence across four regions: Middle East & North Africa, Sub-Saharan Africa, Latin America & Caribbean, and Asia & Pacific.

**Key Findings:**
- Regional structural breaks exist between regions in polarization-conflict dynamics
- Religious Principles and Minority Rights polarization strongly predict conflict
- Different types of polarization affect different regions uniquely
- Mixed-effects modeling reveals significant country and era-level effects

## Repository Structure

```
.
├── data_prep.ipynb           # Data preparation and polarization index creation
├── modelling.ipynb           # Mixed effects modeling and Chow's tests
├── visualisations.ipynb      # Results visualization and robustness checks
├── data/                     # Raw and processed datasets
├── src/                      # Helper functions for analysis
└── assets/figs/             # Generated figures and plots
```

## Data Sources

- **Political Identities**: V-Party Dataset (V-Dem Project) - party stances and ideologies
- **Conflicts**: Uppsala Conflict Data Program (UCDP) - georeferenced conflict events
- **Controls**: World Bank (GDP, Gini), V-Dem (freedom indices, population)

## Methodology

### 1. Data Preparation (`data_prep.ipynb`)

**Polarization Index Creation:**
- Modified Dalton's polarization formula with 'seat share' weights instead of 'vote share'
- Calculate polarization across 15 political manifesto dimensions
- Create conflict intensity (event rate) and severity (death rate) variables

**Key Variables:**
- **Independent**: 15 polarization indices (Anti-Elitism, Religious Principles, Minority Rights, etc.)
- **Dependent**: Conflict event rate and death rate (per 100,000 population, normalized by regime duration)
- **Controls**: GDP per capita, Gini coefficient, population, freedom indices

### 2. Statistical Modeling (`modelling.ipynb`)

**Chow's Test for Structural Breaks:**
- Tests for regional heterogeneity in polarization-conflict relationships
- Identifies significant differences between regional model coefficients

![Chow Test Heatmap](assets/figs/chow's/Chow_Test_Heatmap.pdf)

**Mixed Effects Modeling:**
- Baseline 3-level model: Region → Country → Era
- Region-specific 2-level models: Country → Era
- Accounts for nested data structure and random effects

<p align="center">
  <img src="assets/figs/baseline_model_arch.pdf" alt="Baseline Model" width="45%" />
  <img src="assets/figs/regionwise_model_arch.pdf" alt="Region-wise Model" width="45%" />
</p>


### 3. Visualization & Validation (`visualisations.ipynb`)

**Results Visualization:**
- Regional coefficient comparisons
- Variance decomposition (ICC, R²)
- Global conflict intensity mapping



![Regional Coefficients - Events](assets/figs/region_all_coefficients_events.pdf)
![Regional Coefficients - Deaths](assets/figs/region_all_coefficients_deaths.pdf)


**Robustness Checks:**
- Residual diagnostics (normality, homoscedasticity, linearity)
- Cook's distance outlier detection
- Multicollinearity assessment (VIF)

![Residual Analysis Example](assets/figs/hetro_linearity/Middle_East_and_North_Africa_residuals.pdf)


## Key Results

### Regional Structural Breaks
5 out of 6 regional pairs show significant structural breaks, indicating distinct polarization-conflict dynamics across regions.


![Structural Breaks Network](assets/figs/structural_breaks_network.pdf)


### Significant Polarization Predictors

**Across Multiple Regions:**
- **Religious Principles Polarization**: Positive effect (MENA, Sub-Saharan Africa)
- **Freedom of Expression**: Negative effect (protective factor)
- **Minority Rights Polarization**: Positive effect (Latin America, Asia-Pacific)

**Region-Specific Patterns:**
- **MENA**: LGBT Social Equality (negative), Economic Left-Right (positive)
- **Sub-Saharan Africa**: Immigration polarization effects
- **Latin America**: Rejection of Political Violence (positive)
- **Asia-Pacific**: Gender Equality (negative), Anti-Elitism (negative)

### Model Performance
- **R² Conditional**: 0.73-1.00 across regions (high explanatory power)
- **ICC Values**: Low country/era clustering (<15%), indicating good model fit
- **Heteroscedasticity**: Only Sub-Saharan Africa models show residual heteroscedasticity



## Helper Functions (`src/`)
- `coeff_importance.py`: Regional coefficient visualization
- `cooks_distance.py`: Outlier detection and analysis
- `hetroskedasticity.py`: Residual diagnostics
- `icc_r2.py`: Variance decomposition plots
- `normality.py`: Q-Q plot generation

## Data Files

### Processed Data
- `data/political identities/X.csv`: Polarization indices and controls
- `data/conflicts/Y_conflicts.csv`: Conflict rates by regime
- `data/modelling/*.pkl`: Fitted model objects

### Raw Data Sources
- V-Party Dataset: `data/political identities/V-Dem-CPD-Party-V2.csv`
- UCDP GED: `data/conflicts/ged241-csv.zip`
- World Bank indicators: `data/controls/`

## Citation

```bibtex
@article{padarha2024political,
  title={Party Ideologies and Political Polarization-Driven Conflicts: A Study of the Global South},
  author={Padarha, Shreyansh},
  institution={Oxford Internet Institute, University of Oxford},
  year={2024}
}
```