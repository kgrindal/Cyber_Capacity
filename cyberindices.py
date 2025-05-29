import pandas as pd
import matplotlib.pyplot as plt # Create Correlation Plots Between Indicies
import numpy as np
import statsmodels.api as sm # to add trend line to charts
from scipy.stats import pearsonr # for correlation values


##########################
## Read and Clean CSV Files
##########################

# Read CSV files
harvard_index = pd.read_csv("Raw_Data/harvard_index.csv")
gciv4_index = pd.read_csv('Raw_Data/gciv4_index.csv')
mit_index = pd.read_csv("Raw_Data/mit_index.csv")
oxford_index = pd.read_csv("Raw_Data/oxford_index.csv")
WBData_2020 = pd.read_csv("Raw_Data/WBData_2020.csv")
attack_counts = pd.read_csv("Raw_Data/Dyadic_Country_Attacks.csv")
cybergreen = pd.read_csv("Raw_Data/cybergreen_count.csv")
cyber_exposure = pd.read_csv("Raw_Data/cyber_exposure.csv")

# Combine major scores
harvard_tot = harvard_index[['country', 'score_overall']]
harvard_tot = harvard_tot.rename(columns={'country': 'Country','score_overall':'Harvard_Score'})
gciv4_tot = gciv4_index[['CountryName', 'GCIv4_Tot']].copy()
gciv4_tot = gciv4_tot.rename(columns={'CountryName': 'Country','GCIv4_Tot':'GCIv4_Score'})
oxford_tot = oxford_index[['Country', 'Total_Estimate']].copy()
oxford_tot = oxford_tot.rename(columns={'Total_Estimate':'Oxford_Score'})
mit_tot = mit_index[['Country', 'The Cyber Defense Index']].copy()
mit_tot  = mit_tot.rename(columns={'The Cyber Defense Index':'MIT_Score'})

from functools import reduce

# Merge DataFrames
index_tots = harvard_tot.copy()
dfs_to_merge = [gciv4_tot, oxford_tot, mit_tot]
index_tots = reduce(lambda left, right: pd.merge(left, right, on="Country", how='outer'), dfs_to_merge, index_tots)

index_tots.head()

##########################
## Create Histograms of scores
##########################

# Plot the histogram and get the bar objects
bars = plt.hist(index_tots["GCIv4_Score"], 20)

# Add counts on top of each bar using the bar objects
for i, bar in enumerate(bars[2]):  # bars[2] contains the bar objects
    height = bars[0][i]  # bars[0] contains the bar heights
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
             f'{int(height)}',
             ha='center', va='bottom')

plt.xlabel('GCIv4 Score')
plt.ylabel('Country Frequency')
plt.show() # plots 194 countries

plt.hist(index_tots["GCIv4_Score"],20)
plt.xlabel('GCIv4 Score')
plt.ylabel('Country Frequency')
plt.show()

plt.hist(index_tots["Oxford_Score"])
plt.hist(index_tots["MIT_Score"])
plt.hist(index_tots["Harvard_Score"])

##########################
## Create Correlation Plots Between Indicies
##########################

#clean_data = index_tots.dropna()

# Create Correlation Plots Between Indicies
def add_trendline(x, y, ax):
    # Calculate regression coefficients
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    beta = model.params[1]  # Slope (beta)
    alpha = model.params[0]   # Intercept (alpha)
    
    # Create trend line values
    x_vals = np.array(ax.get_xlim())
    y_vals = alpha + beta * x_vals
    
    # Add the trend line to the plot
    ax.plot(x_vals, y_vals, 'r-', linewidth=2, label='Trend Line')


# Create scatterplot with a label on every point
plt.figure(figsize=(10, 6))
plt.scatter(index_tots['MIT_Score'], index_tots['GCIv4_Score'])
for i, text in enumerate(index_tots['Country']):
    plt.annotate(text, (index_tots['GCIv4_Score'].iloc[i], index_tots['MIT_Score'].iloc[i]))
#add_trendline(index_tots['MIT_Score'], index_tots['GCIv4_Score'], plt.gca())
plt.title('Correlation between GCI and MIT Index')
plt.xlabel('MIT Score')
plt.ylabel('GCI Score')
plt.show()




# Create scatterplot with a label on every point
plt.figure(figsize=(10, 6))
plt.scatter(index_tots['MIT_Score'], index_tots['Harvard_Score'])
for i, text in enumerate(index_tots['Country']):
    plt.annotate(text, (index_tots['MIT_Score'].iloc[i], index_tots['Harvard_Score'].iloc[i]))
#add_trendline(index_tots['MIT_Score'], index_tots['Harvard_Score'], plt.gca())
plt.title('Correlation between MIT and Harvard Index')
plt.xlabel('MIT Score')
plt.ylabel('Harvard Score')
plt.show()

corr, _ = pearsonr(index_tots['MIT_Score'], index_tots['Harvard_Score'])
print('Correlation:', corr)


# Create scatterplot with a label on every point
plt.figure(figsize=(10, 6))
plt.scatter(index_tots['Oxford_Score'], index_tots['GCIv4_Score'])
for i, text in enumerate(index_tots['Country']):
    plt.annotate(text, (index_tots['Oxford_Score'].iloc[i], index_tots['GCIv4_Score'].iloc[i]))
plt.title('Correlation between Oxford and GCI Index')
plt.xlabel('Oxford Score')
plt.ylabel('GCI Score')
plt.show()

################################
## Initial Regression on GCIv4
################################

# Merge DataFrames
from functools import reduce

WBData_2020  = WBData_2020.rename(columns={'CountryName':'Country'})

reg_index = harvard_tot.copy()
dfs_to_merge = [gciv4_tot, oxford_tot, mit_tot, WBData_2020]
reg_index = reduce(lambda left, right: pd.merge(left, right, on="Country", how='outer'), dfs_to_merge, reg_index)

# Subset the merged DataFrame
gci_reg = reg_index[['Country', 'GCIv4_Score', 
                    'Wealth2020', 'Milprcnt2020', 'IntUse2020', 
                    'GovEff2020', 'Enroll2020']].copy()

# View the first few rows
print("First few rows of gci_reg:")
print(gci_reg.head())

# Remove rows with missing values
gci_full_reg = gci_reg.dropna()

# Create a list of column names for easier reference
columns = ['GCIv4_Score', 'Wealth2020', 'Milprcnt2020', 
           'IntUse2020', 'GovEff2020', 'Enroll2020']

# Transform variables as needed
gci_full_reg['exp_GCI'] = np.exp(gci_full_reg[columns[0]])
gci_full_reg['log_Wealth'] = np.log(gci_full_reg[columns[1]])
gci_full_reg['log_Milprcnt'] = np.log(gci_full_reg[columns[2]])
gci_full_reg['inverse_GovEff'] = 1 / gci_full_reg[columns[4]]

# Create the model formula
formula = f"exp_GCI ~ log_Wealth + log_Milprcnt + IntUse2020 + inverse_GovEff + Enroll2020"

# Fit the linear regression model using statsmodels
import statsmodels.api as sm

model = sm.OLS.from_formula(formula, data=gci_full_reg)
regmodel = model.fit()

# Summary of the model
print("\nRegression Model Summary:")
print(regmodel.summary())


################################
## CyberGreen Regression
################################



################################
## Cyberexposure Regression
################################



################################
## Clean World Bank Data and Initial Analysis
################################



