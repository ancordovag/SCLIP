import json
import time
from googletrans import Translator
import yaml
from pathlib import Path
import os

if __name__ == '__main__':

    with open(os.path.join("preprocessing","config.yml"), "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    directory = cfg["coco"]["out_dir"]
    languages = cfg["languages"]

    # source data
    f_json = open(os.path.join(directory, 'en_pairs.json'), mode='r', encoding='utf-8')
    pairs_data = json.load(f_json)

    for lang, code in languages.items():
        if code == "en":
            continue

        path = os.path.join(directory, '{}_pairs.json'.format(code))
        if os.path.isfile(path):
            print("Skipping language {}".format(lang))
            continue
        else:
            w_json = open(path, mode='w', encoding='utf-8')

        print("Translating to {} with code {}".format(lang, code))
        translator = Translator()

        data = []
        for i, pair in enumerate(pairs_data):
            image_id = pair["image_id"]
            caption = pair["caption"]
            translation = translator.translate(caption, dest=code)
            new_caption = translation.text
            item = {'image_id': image_id, 'caption': new_caption}
            if i % 500 == 0:
                print("Debug - Caption: {}  --> Translation: {}".format(caption, new_caption))
            data.append(item)
            time.sleep(0.05)
        f_json.close()

        json_string = json.dumps(data)
        w_json.write(json_string)
        w_json.close()
        time.sleep(0.5)
        print("Language {} finished".format(lang))
    print("Done")
