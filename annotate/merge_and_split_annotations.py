import csv
import os
import random
from typing import List, Tuple

random.seed(0)

team_annotations_path = 'data/team_annotations/'
train_output_path = 'data/train.csv'
validation_output_path = 'data/validation.csv'
validation_proportion = 0.1

team_annotations: List[Tuple[str, str]] = []
for fpath in os.scandir(team_annotations_path):
    with open(fpath.path, 'r', encoding='utf-8', newline='') as f:
        for delim in [',', '\t']:
            reader = csv.reader(f, delimiter=delim)
            for row in reader:
                if len(row) != 2:
                    break
                text, label = row
                team_annotations.append((text, label))
total_size = len(team_annotations)

random.shuffle(team_annotations)
validation_size = int(total_size * validation_proportion)
validation_data = team_annotations[:validation_size]
train_data = team_annotations[validation_size:]

with open(train_output_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(train_data)

with open(validation_output_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(validation_data)
