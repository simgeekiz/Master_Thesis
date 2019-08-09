import os
import json
import json_lines

def load_data(project_folder='/home/aorus/workspaces/simge/Master_Thesis', label_count=2):
    train_data = []
    train_data_path = os.path.join(project_folder, 'Data/thesis_train_data.jsonl')
    with json_lines.open(train_data_path) as reader:
        for obj in reader:
            train_data.append(obj)

    valid_data = []
    valid_data_path = os.path.join(project_folder, 'Data/thesis_valid_data.jsonl')
    with json_lines.open(valid_data_path) as reader:
        for obj in reader:
            valid_data.append(obj)

    test_data = []
    test_data_path = os.path.join(project_folder, 'Data/thesis_test_data.jsonl')
    with json_lines.open(test_data_path) as reader:
        for obj in reader:
            test_data.append(obj)
            
    if label_count == 2:
        for article in train_data:
            for sent in article['sentences']:
                if sent['label'] == 2:
                    sent['label'] = 0

        for article in valid_data:
            for sent in article['sentences']:
                if sent['label'] == 2:
                    sent['label'] = 0

        for article in test_data:
            for sent in article['sentences']:
                if sent['label'] == 2:
                    sent['label'] = 0

    return train_data, valid_data, test_data