# Playground_series_S5E8
A 2 digit final rank on kaggle Playground Series Season 5 Episode 8 , It involves great feature engineering and 2 optuna tuned models and a simple blending approach 


# Kaggle Playground Series S5E8 – Ranked 75/3365 🎉

Led a team of three (from Japan, China, and me) to an exciting **75th place** out of 3,365 in a fierce competition predicting client term deposit subscriptions (target: yes/no), with ROC AUC as the metric.

I focused on extensive feature engineering and trained two hyperparameter-tuned models: **LightGBM** and **CatBoost** (both optimized with Optuna).

The tricky part: many were blending dozens of submissions blindly. To beat this, I kept my models' predictions separate, combined them with the official blended submission using a **weighted average** to preserve true probability distributions. This clever approach pushed us to **rank 28 on the public leaderboard**, surpassing the best existing blend (0.97772 ROC AUC) amidst very tight competition.

Throughout, we tackled class imbalance with SMOTE, incorporated external datasets, and learned from the Kaggle community.  
Working with a diverse, global team made the experience even more rewarding.

Proud of the two-digit rank and the hard-fought, thrilling journey!

#Kaggle #MachineLearning #FeatureEngineering #Optuna #LightGBM #CatBoost #Teamwork
