import pandas as pd

#Data get merged
file_path = "merged_data.xlsx"
df1 = pd.read_excel(file_path)
print(df1)
#Data get cases

file_path = "case_data_R.xlsx"
df2 = pd.read_excel(file_path)
df2.rename(columns={'pavardė': 'Pavardė'}, inplace=True)
print(df2)

# Convert all strings in the 'Pavardė' column to lowercase in both DataFrames
#df1['Pavardė'] = df1['Pavardė'].str.lower()
#df2['Pavardė'] = df2['Pavardė'].str.lower()
# Convert all strings in the 'Pavardė' column to lowercase and strip whitespace in both DataFrames
df1['Pavardė'] = df1['Pavardė'].str.lower().str.strip()
df2['Pavardė'] = df2['Pavardė'].str.lower().str.strip()

#Manual correction

#'blinkienė'="binkienė', 'sakalauskaitė'='sakalauskait', 'girdžis'='girgždis', 'beloborodovienė'='beloborodovi', 'kavaliūnas'='kavaliunas'

#Compare sets

# Ensure both DataFrames have the 'Pavardė' column
if 'Pavardė' in df1.columns and 'Pavardė' in df2.columns:

    # Convert columns to sets for comparison
    set_df1 = set(df1['Pavardė'])
    set_df2 = set(df2['Pavardė'])

    # Compare the sets
    common = set_df1 & set_df2
    only_in_df1 = set_df1 - set_df2
    only_in_df2 = set_df2 - set_df1

    print("Common values in both DataFrames:", common)
    print("Values only in df1:", only_in_df1)
    print("Values only in df2:", only_in_df2)

    # Optionally, compare element-wise and create a comparison DataFrame
    comparison_df = pd.merge(df1, df2, on='Pavardė', how='outer', indicator=True)
    print("\nComparison DataFrame:\n", comparison_df)

else:
    print("Both DataFrames must have the 'Pavardė' column.")

# Set 'Pavardė' as the index for both DataFrames
df1.set_index('Pavardė', inplace=True)
df2.set_index('Pavardė', inplace=True)

# Define a mapping of equivalent names
equivalence_mapping = {
    'blinkienė': 'binkienė',
    'sakalauskaitė': 'sakalauskait',
    'girdžis': 'girgždis',
    'beloborodovienė': 'beloborodovi',
    'kavaliūnas': 'kavaliunas'
}

# Create a new DataFrame to hold the normalized index for df1
df1_normalized = df1.copy()
df1_normalized.index = df1_normalized.index.map(lambda x: equivalence_mapping.get(x, x))

# Create a new DataFrame to hold the normalized index for df2
df2_normalized = df2.copy()
df2_normalized.index = df2_normalized.index.map(lambda x: equivalence_mapping.get(x, x))

# Merge the DataFrames on the index
merged_df = pd.merge(df1_normalized, df2_normalized, left_index=True, right_index=True, how='outer')

# Display the merged DataFrame
print("Merged DataFrame:\n", merged_df)

# Print merged DataFrame to an Excel file
output_file_path = 'final_data.xlsx'
merged_df.to_excel(output_file_path)