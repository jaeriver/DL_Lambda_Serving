import time
import boto3
from transformers import BertTokenizer, TFBertModel

def get_model(model_name, bucket_name):
    s3_resource = boto3.resource('s3')
    s3_bucket = s3_resource.Bucket(bucket_name)
    
    prefix = 'tf/' + model_name
    for object in bucket.objects.filter(Prefix = prefix):
        if object.key == prefix:
            os.makedirs(os.path.dirname('/tmp/' + object.key), exist_ok=True)
            continue;
        bucket.download_file(object.key, '/tmp/' + object.key)
    
    return '/tmp/' + model_name

def lambda_handler(event, context):
    bucket_name = event['bucket_name']
    batch_size = event['batch_size']
    model_name = event['model_name']
    count = event['count']
    arch_type = 'intel'
    
    model_path = get_model(model_name, bucket_name)
    
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = TFBertModel.from_pretrained(model_path)
    sentence = "This is Fake Dataset for testing NLP Tokenizing"
    test_batch = [sentence for i in range(batch_size)]
    encoded_input = tokenizer(test_batch, 
                          return_tensors='tf',
                          padding=True,
                          truncation=True,
                          max_length=512)
    
    time_list = []
    for i in range(count):
        start_time = time.time()
        output = model(encoded_input)
        running_time = time.time() - start_time
        print(f"TF NLP {model_name}-{batch_size} inference latency : ",(running_time)*1000,"ms")
        print(output)
        time_list.append(running_time)
    time_medium = np.median(np.array(time_list))
    return time_medium
