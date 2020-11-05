import torch, os, libfastapi
from flask import Flask, json, request
from flask_cors import CORS
from transformers import ElectraConfig, ElectraTokenizer, ElectraForSequenceClassification

# settings
model_path = 'koelectra-base-v3-sentiment-model'
max_length = 256
labels = ['0', '1']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print('use Torch Device `{}`'.format(device))

config = ElectraConfig.from_pretrained(
    model_path,
    num_labels=len(labels),
    id2label={str(i): label for i, label in enumerate(labels)},
    label2id={label: i for i, label in enumerate(labels)},
)

# load tokenizer
tokenizer = ElectraTokenizer.from_pretrained(
    model_path,
    do_lower_case=False
)

# load & deploy model
model = ElectraForSequenceClassification.from_pretrained(model_path, config=config)
model.to(device)

print('Fine-tuned BertClassificationModel `{}` loaded.'.format(model_path))
print('> Location: `{}`'.format(os.path.abspath(model_path)))

sentiment_analysis_api = Flask(__name__)

# set CORS origins
sentiment_analysis_api.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(sentiment_analysis_api, resources={r'/*': {'origins': '*'}})

# set max content length
sentiment_analysis_api.config['MAX_CONTENT_LENGTH'] = 2048 * 1024 * 1024


def infer(text):
    encoded = tokenizer.encode_plus(
        text,
        max_length=max_length,
        padding='max_length',
        add_special_tokens=True,
        truncation=True
    )

    input_ids = torch.unsqueeze(torch.tensor(encoded['input_ids'], dtype=torch.long).to(device, dtype=torch.long), dim=0)
    attention_mask = torch.unsqueeze(torch.tensor(encoded['attention_mask'], dtype=torch.long).to(device, dtype=torch.long), dim=0)
    token_type_ids = torch.unsqueeze(torch.tensor(encoded['token_type_ids'], dtype=torch.long).to(device, dtype=torch.long), dim=0)
    outputs = model(input_ids, attention_mask, token_type_ids)

    sentiment = 'positive' if int(outputs[0].argmax().data) == 1 else 'negative'
    raw = torch.squeeze(outputs[0]).tolist()
    softmax = torch.squeeze(torch.softmax(outputs[0], 1)).tolist()

    return {
        'sentiment': sentiment,
        'raw': raw,
        'softmax': softmax
    }


@sentiment_analysis_api.route('/sentiment_analysis', methods=['POST'])
def sentiment_analysis():
    print('POST `sentiment()`')
    if request.method == 'POST':
        # print('[POST]', request.values)

        if 'text' in request.values:
            text = request.values['text']
        else:
            text = ''

        inferred = infer(text)
        print(inferred)

        response = sentiment_analysis_api.response_class(
            response=json.dumps(inferred),
            status=200,
            mimetype='application/json',
        )

        response.headers['Access-Control-Allow-Origin'] = '*'

        return response


if __name__ == '__main__':
    sentiment_analysis_api_port = libfastapi.get_stage_values('api')['port']
    sentiment_analysis_api.run(host='0.0.0.0', port=int(sentiment_analysis_api_port), debug=False)
