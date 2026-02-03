import pandas as pd

files = ["train.text", "test.txt", "validation.text"]

all_data = []

for file in files:
    df = pd.read_csv(file)  # already comma separated with headers
    all_data.append(df)

final_df = pd.concat(all_data, ignore_index=True)

final_df.to_csv("emotion.csv", index=False)

print(" emotion.csv created successfully!")
print(final_df.head())
