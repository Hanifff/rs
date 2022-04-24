import chunk
import hashlib
import json


for i in range(1, 1):
    origin_path = "../datasets/netflix/part-0"+str(i)+".json"
    des_path = "../datasets/proc_netflix/pre_part-0"+str(i)+".json"
    with open(origin_path) as f:
        data = json.load(f)

    for item in data:
        item['review_id'] = abs(hash(item['review_id'])) % (10 ** 8)
        item['reviewer'] = abs(hash(item['reviewer'])) % (10 ** 8)

        if item['rating'] != None:
            item['rating'] = int(item['rating'])
        else:
            item['rating'] = 0
    with open(des_path, 'w') as f:
        json.dump(data, f)
