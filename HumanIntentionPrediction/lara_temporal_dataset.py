import pandas as pd
import numpy as np

INPUT = "lara_pose_dataset.csv"
OUTPUT = "lara_temporal_dataset.csv"

WINDOW = 10
MAX_PER_CLASS = 1200
CHUNK_SIZE = 100000

classes = ["Standing", "Walking", "Handling"]

# Read column names first
columns = pd.read_csv(INPUT, nrows=0).columns.tolist()

feature_cols = [
    c for c in columns
    if c not in ["Subject", "Recording", "Frame", "Label"]
]

print("Feature count:", len(feature_cols))
print("Reading dataset in chunks...")

samples = {
    "Standing": [],
    "Walking": [],
    "Handling": []
}

total_windows = 0

for chunk_no, df in enumerate(
    pd.read_csv(INPUT, chunksize=CHUNK_SIZE)
):

    print(f"Processing chunk {chunk_no + 1}...")

    # Process each recording present in this chunk
    for (subject, recording), group in df.groupby(
        ["Subject", "Recording"], sort=False
    ):

        group = group.reset_index(drop=True)

        # Split whenever label changes
        labels = group["Label"].to_numpy()

        start = 0

        for i in range(1, len(group) + 1):

            if i == len(group) or labels[i] != labels[start]:

                segment = group.iloc[start:i]
                label = labels[start]

                if label in classes and len(segment) >= WINDOW:

                    # Non-overlapping temporal windows
                    for j in range(
                        0,
                        len(segment) - WINDOW + 1,
                        WINDOW
                    ):

                        if len(samples[label]) >= MAX_PER_CLASS:
                            break

                        window = segment.iloc[j:j + WINDOW]

                        X = window[feature_cols].to_numpy(
                            dtype=np.float32
                        ).flatten()

                        samples[label].append(
                            [
                                subject,
                                recording,
                                window.iloc[0]["Frame"],
                                label,
                                *X
                            ]
                        )

                        total_windows += 1

                start = i

    # Stop once all classes have enough samples
    if all(len(samples[c]) >= MAX_PER_CLASS for c in classes):
        print("Enough samples collected.")
        break


print("\nCreating temporal dataset...")

rows = []

for label in classes:
    rows.extend(samples[label])

feature_names = [
    f"F{i}"
    for i in range(WINDOW * len(feature_cols))
]

output_columns = [
    "Subject",
    "Recording",
    "StartFrame",
    "Label"
] + feature_names

result = pd.DataFrame(
    rows,
    columns=output_columns
)

result.to_csv(OUTPUT, index=False)

print("\n" + "=" * 50)
print("TEMPORAL DATASET COMPLETE")
print("=" * 50)

print("Samples:", len(result))
print("Features:", len(feature_names))

print("\nClass distribution:")
print(result["Label"].value_counts())

print("\nSaved as:")
print(OUTPUT)
