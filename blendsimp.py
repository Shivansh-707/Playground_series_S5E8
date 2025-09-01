import pandas as pd

# read the two files
df1 = pd.read_csv("Blend.csv")
df2 = pd.read_csv("submission_lightgbm.csv")
df3 = pd.read_csv('submission_catboost.csv')




# take average of y directly
df1["y"] = (df1["y"] + df2['y'] + df3['y']) / 3

# keep only id and y
df1 = df1[["id", "y"]]

# save final file
df1.to_csv("final_submission_lessgo.csv", index=False)
