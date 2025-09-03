# Playground_series_S5E8
A 2 digit final rank on kaggle Playground Series Season 5 Episode 8 , It involves great feature engineering and 2 optuna tuned models and a simple blending approach 


# Kaggle Playground Series S5E8 – Ranked 75/3365 🎉

Led a team of three (from Japan, China, and me) to an exciting **75th place** out of 3,365 in a fierce competition predicting client term deposit subscriptions (target: yes/no), with ROC AUC as the metric.

I focused on extensive feature engineering and trained two hyperparameter-tuned models: **LightGBM** and **CatBoost** (both optimized with Optuna).

The tricky part: Blended Submissions
to nullify the errors made by one file , the simpliest way is to take an average of 2 files , you can either do this by ensembling , or by ofcourse , simply calling y1+y2/2 ,
the later seems less computational heavy and to be honest , easier and faster to play with ... 

now imagine a blend of 30+ files , the problem is , originality was lost ... 
look at it this way , if I let a catboost predict , the target values are actual predictions , logically created my a machine ... right ? 
but the now the same y columns is just averaged out across 30+ files , originality is lost , but they still manage to do well on both the private and the public leaderboard because of one thing ... "ROC AUC metric" .. the metric doesn't care what probablity you give to the buyer / non buyer , all it cares about is the probablity of a buyer should rank above non buyer , this blending case would not exist if it were a binary classification Task of buyer vs non buyer of bank term deposits , 
so anyways , how do you beat blenders at their own game ? 
you beat them by incorporating something that they missed , that's originality , I only blended 3 files , 2 of which were my own model predicition + the top blneded file that time , and I managed to push above all blended code to 0.97773 , but Shakedown was expected , still I am happy to finish in the top 3% ... ready to push for the next challenge 💪 

Throughout, we tackled class imbalance with trying a lot of diffrent things then finally agreeing on bringing only target = 1 rows from the external ( original ) data , my LGBM file won a bronze medal in this compeition as well on the Kaggle code comuunity.  
Working with a diverse, global team made the experience even more rewarding.

Proud of the two-digit rank and the hard-fought, thrilling journey!

#Kaggle #MachineLearning #FeatureEngineering #Optuna #LightGBM #CatBoost #Teamwork
