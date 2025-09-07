import numpy as np
import pandas as pd
import zipfile
import os

if not os.path.exists("Datasets"):
    with zipfile.ZipFile("Datasets.zip", 'r') as zip_ref:
        zip_ref.extractall()
else:
    print("Directory 'Datasets' already exists. Skipping extraction.")


raw_df = pd.read_csv("./Datasets/Food_Inspections_20250413.csv")
