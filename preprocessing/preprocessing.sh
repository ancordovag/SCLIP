# extract data in splits for Europarl (english/spain)
python europarl_preprocessing.py
python europarl_spanish_preprocessing.py
echo "Europarl dataset extraction completed"
# extract data in splits for COCO
python coco_preprocessing.py
echo "Coco dataset extraction completed"
# generate language pairs
python pairs.py
echo "Pairs generated successfully"
# generate multi-language test-set for coco via machine translation
python translate.py
echo "Multilanguage training set done. No errors detected"