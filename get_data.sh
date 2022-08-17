if [ ! $# -eq 1 ];then
   echo "get_data: missing dataset folder \n Usage: get_data.sh [PATH]"
   exit 1
fi

cd $1
wget https://opus.nlpl.eu/download.php?f=Europarl/v8/moses/en-es.txt.zip -O a.zip && unzip a.zip && rm a.zip
wget https://object.pouta.csc.fi/OPUS-Europarl/v8/moses/bg-es.txt.zip -O a.zip && unzip a.zip && rm a.zip

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O a.zip && unzip a.zip && rm a.zip
wget http://images.cocodataset.org/zips/train2017.zip -O a.zip && unzip a.zip && rm a.zip
wget http://images.cocodataset.org/zips/val2017.zip -O a.zip && unzip a.zip && rm a.zip