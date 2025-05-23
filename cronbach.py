import pandas as pd
import pingouin as pg

##########################
## Read and Clean CSV Files
##########################

# Read CSV files
harvard_index = pd.read_csv("Raw_Data/harvard_index.csv")
gciv4_index = pd.read_csv('Raw_Data/gciv4_index.csv')
gciv5_index = pd.read_csv('Raw_Data/gciv5_index.csv')
mit_index = pd.read_csv("Raw_Data/mit_index.csv")
oxford_index = pd.read_csv("Raw_Data/oxford_index.csv")

########################
# Harvard
########################

# Coding categories is based on Code Book linked to here: https://dataverse.harvard.edu/file.xhtml?persistentId=doi:10.7910/DVN/LT55JY/HQMNOH&version=3.0

capabilities = harvard_index[['score_capabilities', 'score_surveillance', 'score_defense', 'score_control', 'score_intelligence', 'score_commercial', 'score_offense', 'score_norms']]
b = pg.cronbach_alpha(data=capabilities)
b

# score_overall	is leftover?
intent = harvard_index[['score_intent', 'intent_surveillance','intent_defense','intent_control', 'intent_intelligence', 'intent_commercial','intent_offense','intent_norms']]
c = pg.cronbach_alpha(data=intent)
c

#Cyber Power Scores: score_captint capint_surveillance	capint_defense	capint_control	capint_intelligence	capint_commercial	capint_offense	capint_norms
#Capability Indicators: 
#laws	web_alexa	news_alexa	removal_google	freedom_net	infocomm_imp	patent_application	patent_app_capita	broadband_speed	mobile_speed	ecommerce	ecommerce_capita	state_attack	attack_objective	attack_surveillance	attack_control	attack_intelligence	attack_commercial	attack_offense	tech_firm	tech_export	human_capital	cybermil_people	cyber_firm	computer_infection	mobile_infection	socials_use	internet_use	surveillance_firm	shodan	military_strategy	cyber_command	CERTS	multi_agreement	bilat_agreement	softpower	ITU


# Intent Scores Construction


########################
# GCI v4 and GCI v5
########################

gciv4_cat = gciv4_index[['Legal','Tech','Org','CapDev','Coop']]
a = pg.cronbach_alpha(data=gciv4_cat)

gciv4_subcat1 = gciv4_index[['Legal1','Legal2']]
gciv4_subcat2 = gciv4_index[['Tech1','Tech2','Tech3','Tech4']]
gciv4_subcat3 = gciv4_index[['Org1','Org2','Org3']]
gciv4_subcat4 = gciv4_index[['CapDev1','CapDev2','CapDev3','CapDev4','CapDev5','CapDev6']]
gciv4_subcat5 = gciv4_index[['Coop1','Coop2','Coop3','Coop4','Coop5']]

b = pg.cronbach_alpha(data=gciv4_subcat1)
c = pg.cronbach_alpha(data=gciv4_subcat2)
d = pg.cronbach_alpha(data=gciv4_subcat3)
e = pg.cronbach_alpha(data=gciv4_subcat4)
f = pg.cronbach_alpha(data=gciv4_subcat5)

# Create a table of Cronbach's alpha values
alpha_table = pd.DataFrame({
    'Subcategory': ['Overall',
                   'Subcategory 1',
                   'Subcategory 2',
                   'Subcategory 3',
                   'Subcategory 4'],
    'Cronbach Alpha': [a, b, c, d, e]
})

# Display the table
alpha_table.set_index('Subcategory').T.reset_index().rename(columns={'index': ''})
alpha_table = alpha_table.round(3)  # Round to 3 decimal places for better readability

print("Cronbach's Alpha Values:")
print(alpha_table)

# GCIv5
gciv5_cat = gciv5_index[['Legal Measures','Technical Measures','Organization Measures','Capacity Development','Cooperation Measures']]
pg.cronbach_alpha(data=gciv5_cat)

########################
# MIT
########################

mit_cat = mit_index[['1. Critical infrastructure','2. Cybersecurity resources','3. Organizational capacity','4. Policy commitment']]
a = pg.cronbach_alpha(data=mit_cat)

mit_subcat1 = mit_index[['1.1 Telecom infrastructure','1.2 Data centers','1.3 Secure servers','1.4 Perceived robustness','1.5 Critical infrastructure comprehensiveness']]
mit_subcat2 = mit_index[['2.1 Digital security capacity','2.2 Data protection','2.3 Security tools and infrastructure effectiveness']]
mit_subcat3 = mit_index[['3.1 E-participation','3.2 Government AI readiness','3.3 Strategic intent','3.4 Industry standards and practices']]
mit_subcat4 = mit_index[['4.1 Regulatory quality','4.2 Government effectiveness','4.3 Business perceptions of regulatory robustness','4.4 Cybersecurity framework comprehensiveness']]

b = pg.cronbach_alpha(data=mit_subcat1)
c = pg.cronbach_alpha(data=mit_subcat2)
d = pg.cronbach_alpha(data=mit_subcat3)
e = pg.cronbach_alpha(data=mit_subcat4)

# Create a table of Cronbach's alpha values
alpha_table = pd.DataFrame({
    'Subcategory': ['Overall',
                   'Subcategory 1',
                   'Subcategory 2',
                   'Subcategory 3',
                   'Subcategory 4'],
    'Cronbach Alpha': [a, b, c, d, e]
})

# Display the table
alpha_table.set_index('Subcategory').T.reset_index().rename(columns={'index': ''})
alpha_table = alpha_table.round(3)  # Round to 3 decimal places for better readability

print("Cronbach's Alpha Values:")
print(alpha_table)

########################
# Oxford
########################

oxford_cat = oxford_index[['D1_Estimate','D2_Estimate','D3_Estimate','D4_Estimate','D5_Estimate']]
a = pg.cronbach_alpha(data=oxford_cat)

oxford_subcat1 = oxford_index[['NatStrat1','NatStrat2','NatStrat3','IR1','IR2','IR3','IR4','CritProt1','CritProt2','CritProt3','CrisisMgmt','CrisisDef1','CrisisDef2','CrisisDef3','CommRed']]
oxford_subcat2 = oxford_index[['MindsetGov','MindsetPriv','MindsetUser','TrustSec1','TrustSec2','TrustSec3','PIIUser','ReportMech1','MassMedia']]
oxford_subcat3 = oxford_index[['AwarnessProg','AwarnessExec','EduProv','EduAdmin','FrameProv','FrameUptake']]
oxford_subcat4 = oxford_index[['LegalFrame1','LegalFrame2','LegalFrame3','LegalFrame4','LegalFrame5','LegalFrame6','LegalFrame7','LegalFrame8','CrimJust1','CrimJust2','CrimJust3','FormalCoop','InformalCoop']]
oxford_subcat5 = oxford_index[['Standard1','Standard2','Standard3','InfrRes','SoftQual','TechControl','CryptoControl','CyberMarket1','CyberMarket2','Disclosure']]

b = pg.cronbach_alpha(data=oxford_subcat1)
c = pg.cronbach_alpha(data=oxford_subcat2)
d = pg.cronbach_alpha(data=oxford_subcat3)
e = pg.cronbach_alpha(data=oxford_subcat4)
f = pg.cronbach_alpha(data=oxford_subcat5)

# Create a table of Cronbach's alpha values
alpha_table = pd.DataFrame({
    'Subcategory': ['Overall',
                   'Dimension 1',
                   'Dimension 2',
                   'Dimension 3',
                   'Dimension 4',
                   'Dimension 5'],
    'Cronbach Alpha': [a, b, c, d, e, f]
})

# Display the table
alpha_table.set_index('Subcategory').T.reset_index().rename(columns={'index': ''})
alpha_table = alpha_table.round(3)  # Round to 3 decimal places for better readability

print("Cronbach's Alpha Values:")
print(alpha_table)