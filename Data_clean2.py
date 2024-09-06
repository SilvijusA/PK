import pandas as pd

#Data merged get
file_path = "merged_data.xlsx"
df2 = pd.read_excel(file_path)
print(df2)

#get SAV data below does not handle lithuanian charachters well therefore it was dropped R is to be used instead
"""
import pandas as pd
import savReaderWriter

# Define file path
file_path = "TX_tyrimo_darbinis_last.sav"

# Use savReaderWriter to read the .sav file without date conversion
with savReaderWriter.SavReader(file_path) as reader:
    records = reader.all()  # Read all records
    header = []
    for s in reader.header:
        # Try decoding the header, and if it fails, leave it as is
        try:
            header.append(s.decode('utf-8'))
        except AttributeError:  # If it's already a string
            header.append(s)
        except UnicodeDecodeError:  # If decoding fails
            header.append(s.decode('latin1'))  # Try 'latin1' encoding as fallback

# Create a DataFrame from the records and header
df = pd.DataFrame(records, columns=header)

# Display the first few rows of the dataframe
print(df.head())


df.to_excel('case_data2.xlsx', index=False)
"""

#Use R

import subprocess
import os

# Define paths
r_script_path = "C:\\Users\\Legion\\Desktop\\pk Aurelija\\Open_sav.R"
rscript_path = "C:\\Program Files\\R\\R-4.2.1\\bin\\Rscript.exe"  # Correct path to Rscript
working_directory = "C:\\Users\\Legion\\Desktop\\pk Aurelija"

# Change the working directory
os.chdir(working_directory)

# Run the R script
result = subprocess.run([rscript_path, r_script_path], capture_output=True, text=True)

# Print the output and errors
print("Output:\n", result.stdout)
print("Errors:\n", result.stderr)
