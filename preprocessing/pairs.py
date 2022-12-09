from os import listdir
from os.path import isfile, join
import json
import yaml
import os

def create_pairs():
    with open(os.path.join("preprocessing", "config.yml"), "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    coco_config = cfg["coco"]
    images_directory = coco_config["image_dir"]
    amount_pairs = coco_config["pairs"]
    annotation_directory = os.path.join(coco_config["caption_dir"], 'captions_val2017.json')

    f_val = open(annotation_directory)
    val_data = json.load(f_val)

    onlyfiles = [f for f in listdir(images_directory) if isfile(join(images_directory, f))]

    # Creating Validation file
    count = 0
    data = []
    for element in val_data["annotations"]:
        caption = element["caption"]
        for z in range(6,8):
            zeros = '0'*z
            image_id = zeros + str(element["image_id"]) + '.jpg'        
            if image_id in onlyfiles:
                item = {'image_id': image_id, 'caption': caption}
                data.append(item)
                onlyfiles.remove(image_id)
                count += 1
                break
        if count >= amount_pairs:
            break
            
    f_val.close()
    print("Number of pairs: {}".format(count))

    pairs_file = open(os.path.join(coco_config["out_dir"], 'en_pairs.json'), mode='w', encoding='utf-8')
    json_string = json.dumps(data)
    pairs_file.write(json_string)
    pairs_file.close()
    
if __name__ == "__main__":
    create_pairs()