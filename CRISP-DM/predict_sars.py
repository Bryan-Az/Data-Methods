import pandas as pd
import pickle
from outbreak_data import authenticate_user, outbreak_data
authenticate_user.authenticate_new_user()


geo_isos = ['BOL', 'BRA', 'CHL']
# sars metadata 1: cases increases
cases_numIncrease = outbreak_data.cases_by_location(geo_isos, pull_smoothed=True)
print('ISOS accessed cases successfully, with shape: ', cases_numIncrease.shape)
# sars metadata 2: lineages prevalences
lineage_prevalences = []
for loc in geo_isos:
    locDf = outbreak_data.prevalence_by_location(loc, other_threshold=0.85)
    locDf['location'] = [loc] * len(locDf)
    lineage_prevalences.append(locDf)
lineage_prevalences = pd.concat(lineage_prevalences)
print('Lineage prevalences accessed successfully, with shape: ', lineage_prevalences.shape)

#sars metadata 3: lineage mutations
# select good candidates for mutation information
interesting_lineages = ['xbb.1.5.76', 'xbb.1.5']
lineage_mutations = outbreak_data.lineage_mutations(pango_lin=interesting_lineages)
print('Lineage names used to access lineage mutations successfully, with shape: ', lineage_mutations.shape)
print('All necessary data gathered from the Outbreak API! Now moving on to preprocessing...')

# Processing the Data
# Selecting most recent 5000 datapoints to plot visual of lineage, as there is too  many to plot
plot_lineage_prevalences=lineage_prevalences.sort_values('date')[-5000:]
# Calculates the cumulative sum of lineages across time, across all locations - in the past 5000 recent datapoints
plot_lineage_prevalences['prevalence_cumSum'] = plot_lineage_prevalences.groupby(['date', 'lineage'])['prevalence_rolling'].cumsum()
most_recent_lineages = plot_lineage_prevalences.loc[:,['prevalence_cumSum', 'lineage']].groupby('lineage').mean().sort_values('prevalence_cumSum')[-20:].index.to_list()
# Selecting top 20 most prevalent in the most recent datapoints to limit visual complexity
plot_recent_lineage_prevalences = plot_lineage_prevalences.where(plot_lineage_prevalences.lineage.apply(lambda x: x in most_recent_lineages)).dropna(how='all')
plot_recent_lineage_prevalences = plot_recent_lineage_prevalences.reset_index(drop=True)
#splitting the dataset into lineage_defined and lineage_undefined
lineage_undefined = plot_recent_lineage_prevalences.where(plot_recent_lineage_prevalences.lineage == 'other').dropna(how='all')
lineage_defined = plot_recent_lineage_prevalences.where(plot_recent_lineage_prevalences.lineage != 'other').dropna(how='all')
# For the lineage_defined dataset, we can go ahead and merge it with the mutations data
sars_epi_viro = pd.merge(lineage_defined, lineage_mutations, on = 'lineage', suffixes=('_cases','_mutations'))
# The _score, admin1 columns are not meaningful and could be dropped. (it is used by the API team)
cases_numIncrease = cases_numIncrease.drop('_score', axis=1)
cases_numIncrease = cases_numIncrease.drop('admin1', axis=1)
# _id column contains the location name, and this is the only unique piece of info 
# feature can be re-engineered to location & then merged with the sars_epi_cov dataset
cases_numIncrease['location'] = cases_numIncrease._id.apply(lambda x: x[:3])
cases_numIncrease = cases_numIncrease.drop('_id', axis = 1)
# Finally, we constructed the training/test dataset for a M.L/A.I model
# It should contain all necessary features for this project
sars_epi_viro = pd.merge(sars_epi_viro, cases_numIncrease, on=['location', 'date'])
print("Full corpus created, with shape: ", sars_epi_viro.shape)
#Encoding the categorical variables in the data for the model
categs = ['date', 'lineage', 'location', 'mutation', 'gene', 'ref_aa', 'alt_aa', 'codon_end', 'type', 'change_length_nt']
for col in categs:
    sars_epi_viro[col] = sars_epi_viro[col].astype('category')
sars_epi_viro_encoded = pd.get_dummies(sars_epi_viro, columns=categs, drop_first=True)
# Implementing the Model by Loading the Pickle Object
with open('sars_lasso_model.pkl', 'rb') as file:
    lasso_model = pickle.load(file)
print('Model loaded, making predictions...')
#this will ignore an error that checks that the names and columns are in the exact order as training data
#since the default configuration (with ex. keys in ISO and Lineages) was used to train
#this should not be a problem, and is likely due to the special characters
#that are used in marking genetics data
if hasattr(lasso_model, 'feature_names_in_'):
    lasso_model.feature_names_in_ = None
predictions = lasso_model.predict(sars_epi_viro_encoded.drop('prevalence_rolling', axis=1))
predictions = pd.Series(predictions)
predictions.to_csv('sars_predictions.csv', index=False)
print('Predictions saved to "sars_predictions.csv"')