import os
import json

def load_data(project_folder='/home/aorus/workspaces/simge/Master_Thesis', shuffle_num=1, label_count=2):

    data_path = os.path.join(project_folder, 'Data/thesis_data_shuffle-' + str(shuffle_num) + '_.json')
    with open(data_path) as reader:
        data_ = json.load(reader)
        
    train_data = data_['data']['train']
    valid_data = data_['data']['valid']
    test_data = data_['data']['test']
            
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

    return train_data, valid_data, test_data, data_['metadata']