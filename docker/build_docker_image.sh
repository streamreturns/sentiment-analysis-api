#/bin/bash
tar cjpvf filesystem.tar.bz2 --directory filesystem .
tar cjpvf koelectra-base-v3-sentiment-model.tar.bz2 koelectra-base-v3-sentiment-model
docker build --tag sentiment-analysis-api .
