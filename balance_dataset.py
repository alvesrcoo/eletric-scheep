import pandas as pd
import numpy as np

# Load unbalanced dataset
# Verify if the dataset is separated with "," or "\t"
dataframe = pd.read_csv('unbalanced_dataset.csv', sep=',')

# Create a dataframe with Deletions
# Choose n for the number of items or frac for a fraction of items
df_deletion = dataframe[dataframe.Variant == "Deletion"].sample(n=5)

# Create a dataframe with NoVariants
df_novariant = dataframe[dataframe.Variant == "NoVariant"].sample(n=5)

# Create a dataframe with SNVs
df_snv = dataframe[dataframe.Variant == "SNV"].sample(n=5)

dataframeok = pd.concat([df_deletion,df_snv,df_novariant], ignore_index = True)

export_csv = dataframeok.to_csv(r'balanced_dataset.csv', index = None, header=True)