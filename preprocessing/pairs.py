from os import listdir
from os.path import isfile, join
from PIL import Image
import json
import yaml
import os


with open("config.yml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

coco_config = cfg["coco"]
images_directory = coco_config["image_dir"]
annotation_directory = os.path.join(coco_config["caption_dir"], 'captions_val2017.json')

f_val = open(annotation_directory)
val_data = json.load(f_val)

onlyfiles = [f for f in listdir(images_directory) if isfile(join(images_directory, f))]

# Creating Validation file
count = 0
data = []
for element in val_data["annotations"]:
    image_id = '000000' + str(element["image_id"]) + '.jpg'
    caption = element["caption"]
    if image_id in onlyfiles:
        item = {'image_id': image_id, 'caption': caption}
        data.append(item)
        onlyfiles.remove(image_id)
        count += 1
        # if 52 < count < 57:
        #     print(caption)
        #     image = Image.open(os.path.join(images_directory, image_id))
        #     image.show()
        if count >= 100:
            break
f_val.close()
print("Number of pairs: {}".format(count))

pairs_file = open(os.path.join(coco_config["out_dir"], 'en_pairs.json'), mode='w', encoding='utf-8')
json_string = json.dumps(data)
pairs_file.write(json_string)
pairs_file.close()