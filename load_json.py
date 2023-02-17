import json
 
def load_json(path):
    with open(path,'r',encoding = 'utf-8') as fp:
        data = json.load(fp)
    return data['images']