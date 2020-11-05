# Sentiment Analysis API
Deep Learning based Sentiment Analysis

## Usage (Python 3)

### Launch Docker Image
```
docker run --rm -p 65281:65281 sentiment-analysis-api python sentiment_analysis_api_server.py
```

### Import Library
```
import requests
```

### Set API Host
```
api_host = 'fastapi.kr'
api_port = '1128'
```

### [Example] Positive Sentiment
- **Request**
```
text = '봄날에 입고가기 좋아요'
r = requests.post('http://{api_host}:{api_port}/sentiment_analysis'.format(api_host=api_host, api_port=int(api_port)), data={'text': text})
```

- **Response**
```
{'raw': [-2.321743965148926, 2.9482204914093018],
 'sentiment': 'positive',
 'softmax': [0.005117471795529127, 0.9948825836181641]}
```

### [Example] Negative Sentiment
- **Request**
```
text = '재질이 거칠어요'
r = requests.post('http://{api_host}:{api_port}/sentiment_analysis'.format(api_host=api_host, api_port=int(api_port)), data={'text': text})
```

- **Response**
```
{'raw': [3.3003969192504883, -3.5723228454589844],
 'sentiment': 'negative',
 'softmax': [0.9989653825759888, 0.0010345850605517626]}
```

### [Example] Bulk Inference
This example analyzes `text` column in the `target_df`.

- **Define Bulk Inference Method `analyze_sentiment`**
```
def analyze_sentiment(row):
    text = prettify_string(row['text'])
    
    r = requests.post('http://{api_host}:{api_port}/sentiment_analysis'.format(api_host=api_host, api_port=int(api_port)),
                      data={
                          'text': text
                      })
    
    if r.ok:
        resp = r.json()
        row['sentiment'] = resp['sentiment']
        row['raw'] = numpy.max(resp['raw'])
        row['softmax'] = numpy.max(resp['softmax'])
    else:
        row['sentiment'] = 'API error'
        row['raw'] = 'API error'
        row['softmax'] = 'API error'    
        
    return row
```

- **Bulk Inference**
```
target_df = target_df.apply(lambda x: analyze_sentiment(x), axis=1)
```

### Supported Languages

Korean (한국어)

---

- **Maintainer**
Byeongho Kang (byeongho.kang@yahoo.com)
- **GitHub**
https://github.com/streamreturns
