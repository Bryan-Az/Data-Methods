This folder contains the Jupyter notebook I used to analyze the Coronavirus using CRISP-DM methodology.
The medium article to accompany the notebook can be found here: https://medium.com/@bryanambzam/an-epidemiological-analysis-of-sars-cov-2-using-pycaret-ce6fe1d7c178
Finally, the model produced in the notebook was saved in this directory.

To deploy the model to make predictions on the localized country (Bolivia, Brazil, Chile) data for Sars-CoV-2, using the most up-to-date
data taken from the Outbreak.API (you will need to authenticate), you will need to:

1. CD into the CRISP-DM directory of this repo
2. Make sure you have installed all packages used in the predict_sars.py file
3. Run the command "python predict_sars.py"
4. Authenticate at the URL when prompted

