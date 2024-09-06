import pandas as pd

#Data cleaning for keys
file_path = "keys.xlsx"
df2 = pd.read_excel(file_path,skiprows=[0, 1, 2,3])
# Drop empty columns (columns where all values are NaN)
df2 = df2.dropna(axis=1, how='all')
df2 = df2.drop('Unnamed: 11', axis=1)

# Convert 'N°' column to numeric, forcing errors to NaN
df2['Vizito Nr.'] = pd.to_numeric(df2['Vizito Nr.'], errors='coerce')

# Drop rows where 'N°' column is NaN
df2 = df2.dropna(subset=['Vizito Nr.'])

df2 = df2.rename(columns={'TPS PRLVT / VISITE': 'TPS PRLVT_VISITE'})

print(df2)
print(df2.columns)
df2.to_excel('cleaned_data2.xlsx', index=False)


#Data cleaning for PK data


# Load the Excel file into a pandas DataFrame
file_path = "MMF Aurelija_dosage 20191010_V0.02.xls"
#df = pd.read_excel(file_path)
df = pd.read_excel(file_path, skiprows=[0, 1, 2, 3])

# Rename columns
df.columns = ['N°', 'CODE PATIENT', 'TPS PRLVT_VISITE', 'NB TUBES', 'DATE TUBE', 'NATURE PRLVT', 'ANALYSE', 'LOCALISATION_BOITE', 'COMMENTAIRE RECEPTION', 'RESULTAT', 'UNITE']

# Display the DataFrame
print(df.head(20))

# Assuming your DataFrame is named 'df'
print(df.columns)

# Convert 'N°' column to numeric, forcing errors to NaN
df['N°'] = pd.to_numeric(df['N°'], errors='coerce')

# Drop rows where 'N°' column is NaN
df = df.dropna(subset=['N°'])
print(df)

# Export the cleaned DataFrame to an Excel file
df.to_excel('cleaned_data.xlsx', index=False)

#Merger

# Assuming df and df2 are already loaded
df = df.set_index('TPS PRLVT_VISITE')
df2 = df2.set_index('TPS PRLVT_VISITE')

# Merge the two DataFrames on the index (left join, inner join, etc. depending on your need)
merged_df = df.merge(df2, left_index=True, right_index=True, how='inner')  # You can change 'how' to 'left', 'right', or 'outer' as needed

# Print merged DataFrame to an Excel file
output_file_path = 'merged_data.xlsx'
merged_df.to_excel(output_file_path)

print(f'Merged data has been exported to {output_file_path}')
