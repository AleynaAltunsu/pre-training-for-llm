# Dataset'i yükle
import datasets

dataset = datasets.load_dataset(
    "parquet", 
    data_files="./data/preprocessed_dataset.parquet", 
    split="train"
)
print(dataset)

# Dataset'i 10 parçaya ayır
dataset = dataset.shard(num_shards=10, index=0)
print(dataset)

# Tokenizer'ı yükle ve test et
from transformers import AutoTokenizer
model_path_or_name = "./models/SOLAR-10.7B-v1.0"
tokenizer = AutoTokenizer.from_pretrained(
    model_path_or_name, 
    use_fast=False
)
print(tokenizer.tokenize("I'm a short sentence"))

# Tokenizasyon için yardımcı fonksiyon oluştur
def tokenization(example):
    # Tokenize işlemi
    tokens = tokenizer.tokenize(example["text"])
    # Token'ları ID'lere çevir
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    # Başına <bos>, sonuna <eos> ekle
    token_ids = [
        tokenizer.bos_token_id] \
        + token_ids \
        + [tokenizer.eos_token_id
    ]
    # input_ids ve num_tokens sütunlarını ekle
    example["input_ids"] = token_ids
    example["num_tokens"] = len(token_ids)
    return example

# Dataset'i tokenize et
dataset = dataset.map(tokenization, load_from_cache_file=False)
print(dataset)

# Örnek bir veri kontrolü
sample = dataset[3]
print("text", sample["text"][:30]) 
print("\ninput_ids", sample["input_ids"][:30])
print("\nnum_tokens", sample["num_tokens"])

# Dataset'teki toplam token sayısını hesapla
import numpy as np
print(np.sum(dataset["num_tokens"]))

# input_ids'i birleştir
input_ids = np.concatenate(dataset["input_ids"])
print(len(input_ids))

# Ekstra token'ları çıkart ve reshape işlemi
max_seq_length = 32
total_length = len(input_ids) - len(input_ids) % max_seq_length
print(total_length)
input_ids = input_ids[:total_length]
print(input_ids.shape)
input_ids_reshaped = input_ids.reshape(-1, max_seq_length).astype(np.int32)
print(input_ids_reshaped.shape)
print(type(input_ids_reshaped))

# Hugging Face Dataset formatına dönüştür
input_ids_list = input_ids_reshaped.tolist()
packaged_pretrain_dataset = datasets.Dataset.from_dict(
    {"input_ids": input_ids_list}
)
print(packaged_pretrain_dataset)

# Paketlenmiş dataset'i diske kaydet
packaged_pretrain_dataset.to_parquet("./data/packaged_pretrain_dataset.parquet")