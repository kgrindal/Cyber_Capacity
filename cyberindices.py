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

def plot_histogram(data_column, title):
    plt.figure(figsize=(6, 4))
    bars = plt.hist(index_tots[data_column], 20)
    
    # Add counts on top of each bar
    for i, bar in enumerate(bars[2]):
        height = bars[0][i]
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                 f'{int(height)}',
                 ha='center', va='bottom')
    
    plt.xlabel('Score')
    plt.ylabel('Country Frequency')
    #plt.title(title)
    
plot_histogram("GCIv4_Score", "GCIv4 Score Distribution")
plt.tight_layout() 
plt.savefig('jpeg/gciv4_freq.jpeg')

def plot_histogram(data_column, title):
    plt.figure(figsize=(6, 4))
    bars = plt.hist(index_tots[data_column], 8)
    
    # Add counts on top of each bar
    for i, bar in enumerate(bars[2]):
        height = bars[0][i]
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                 f'{int(height)}',
                 ha='center', va='bottom')
    
    plt.xlabel('Score')
    plt.ylabel('Country Frequency')
    #plt.title(title)

plot_histogram("MIT_Score", "MIT Score Distribution")
plt.tight_layout() 
plt.savefig('jpeg/mit_freq.jpeg')

plot_histogram("Oxford_Score", "Oxford Score Distribution")
plt.tight_layout() 
plt.savefig('jpeg/oxford_freq.jpeg')

plot_histogram("Harvard_Score", "Harvard Score Distribution")
plt.tight_layout() 
plt.savefig('jpeg/harvard_freq.jpeg')


##########################
## Create Correlation Plots Between Indicies - TEST
##########################


def create_clean_data(df, var1, var2):
    """Filter DataFrame to keep only complete rows for two variables"""
    return df.dropna(subset=[var1, var2])

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


def plot_correlation(df, var_x, var_y, title):
    """Plot a correlation scatterplot with trendline and country labels"""
    plt.figure(figsize=(6, 4))
    
    # Filter data
    clean_data = create_clean_data(df, var_x, var_y)
    
    # Create scatterplot
    scatter = plt.scatter(clean_data[var_x], clean_data[var_y], 
                         alpha=0.5, s=40, c='blue', edgecolors='black')
    
    # Add trendline
    add_trendline(clean_data[var_x], clean_data[var_y], plt.gca())
    
    # Add country labels with offset to prevent overlap
    indexer = np.arange(len(clean_data))
    for i, (x_val, y_val, country) in enumerate(
        zip(clean_data[var_x], clean_data[var_y], clean_data['Country'])
    ):
        offset_x = .15
        offset_y = .2
        # # Calculate horizontal offset based on position
        # if indexer[i] < len(indexer) / 2:
        #     offset_x = -0.15
        # else:
        #     offset_x = +0.15

        plt.annotate(
            text=country,
            xy=(x_val + offset_x, y_val + offset_y),
            ha='left' if offset_x == -0.15 else 'right',
            va='bottom',
            fontsize=8,
            color='black'
        )
    
    plt.title(title)
    plt.xlabel(var_x)
    plt.ylabel(var_y)
    plt.tight_layout()
    
    # Save the plot with filename in format "corr_varxvary.jpeg"
    filename = f"jpeg/corr_{var_x.replace(' ', '_')}_{var_y.replace(' ', '_')}.jpeg"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()  # Close the figure to free up memory

# Example usage:
plot_correlation(index_tots, 
                 'MIT_Score', 
                 'GCIv4_Score',
                 'Correlation between MIT and GCIv4 Index')

plot_correlation(index_tots,
                 'MIT_Score',
                 'Harvard_Score',
                 'Correlation between MIT and Harvard Index')

plot_correlation(index_tots,
                 'Oxford_Score',
                 'GCIv4_Score',
                 'Correlation between Oxford and GCI Index')

plot_correlation(index_tots,
                 'Harvard_Score',
                 'GCIv4_Score',
                 'Correlation between Harvard and GCI Index')


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



