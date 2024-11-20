# Model eğitimi için gerekli kütüphaneler ve ayarlar
import warnings
warnings.filterwarnings('ignore')
import torch
from transformers import AutoModelForCausalLM

# 1. Model Yükleme
# Daha önce oluşturulan upscaled modeli yükler.
pretrained_model = AutoModelForCausalLM.from_pretrained(
    "./models/TinySolar-308m-4k-init",
    device_map="cpu", 
    torch_dtype=torch.bfloat16,
    use_cache=False,
)
pretrained_model

# 2. Veri Seti Yükleme
# Özel veri seti sınıfı tanımlanır ve veriler PyTorch ile uyumlu hale getirilir.
import datasets
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, args, split="train"):
        # Veri setini başlatır.
        self.args = args
        self.dataset = datasets.load_dataset(
            "parquet",
            data_files=args.dataset_name,
            split=split
        )

    def __len__(self):
        # Veri setindeki örnek sayısını döndürür.
        return len(self.dataset)

    def __getitem__(self, idx):
        # Belirtilen bir indeksteki örneği döndürür.
        input_ids = torch.LongTensor(self.dataset[idx]["input_ids"])
        labels = torch.LongTensor(self.dataset[idx]["input_ids"])
        return {"input_ids": input_ids, "labels": labels}

# 3. Eğitim Argümanlarının Yapılandırılması
# Eğitim için gereken hiperparametreler ayarlanır.
from dataclasses import dataclass, field
import transformers

@dataclass
class CustomArguments(transformers.TrainingArguments):
    dataset_name: str = field(default="./parquet/packaged_pretrain_dataset.parquet")
    num_proc: int = field(default=1)
    max_seq_length: int = field(default=32)
    seed: int = field(default=0)
    optim: str = field(default="adamw_torch")
    max_steps: int = field(default=30)
    per_device_train_batch_size: int = field(default=2)
    learning_rate: float = field(default=5e-5)
    weight_decay: float = field(default=0)
    warmup_steps: int = field(default=10)
    lr_scheduler_type: str = field(default="linear")
    gradient_checkpointing: bool = field(default=True)
    dataloader_num_workers: int = field(default=2)
    bf16: bool = field(default=True)
    gradient_accumulation_steps: int = field(default=1)
    logging_steps: int = field(default=3)
    report_to: str = field(default="none")

# Argümanlar ve veri seti yüklenir.
parser = transformers.HfArgumentParser(CustomArguments)
args, = parser.parse_args_into_dataclasses(args=["--output_dir", "output"])
train_dataset = CustomDataset(args=args)

# Veri seti şekli kontrol edilir.
print("Input shape: ", train_dataset[0]['input_ids'].shape)

# 4. Eğitim ve Kayıp Takibi
# Özel bir callback ile kayıp değerlerini izlemek için ayarlar yapılır.
from transformers import Trainer, TrainingArguments, TrainerCallback

class LossLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            self.logs.append(logs)

    def __init__(self):
        self.logs = []

loss_logging_callback = LossLoggingCallback()

# Hugging Face Trainer nesnesi oluşturulur ve eğitim başlatılır.
trainer = Trainer(
    model=pretrained_model, 
    args=args, 
    train_dataset=train_dataset, 
    eval_dataset=None,
    callbacks=[loss_logging_callback] 
)

trainer.train()

# 5. Ara Kontrol Noktasını Değerlendirme
# Eğitim sırasında kaydedilen bir kontrol noktasını yükleyip değerlendirme yapılır.
from transformers import AutoTokenizer, TextStreamer, AutoModelForCausalLM

model_name_or_path = "./models/upstage/output/checkpoint-10000"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model2 = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,    
)

prompt = "I am an engineer. I love"
inputs = tokenizer(prompt, return_tensors="pt").to(model2.device)

streamer = TextStreamer(
    tokenizer, 
    skip_prompt=True, 
    skip_special_tokens=True
)

outputs = model2.generate(
    **inputs, 
    streamer=streamer, 
    use_cache=True, 
    max_new_tokens=64,     
    do_sample=True,
    temperature=1.0,
)
