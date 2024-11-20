# Önemsiz uyarıları göz ardı et (örn. eskiyen özellik uyarıları)
import warnings
warnings.filterwarnings('ignore')

# Çoğaltılabilirlik için rastgele tohum değerini sabitle
import torch

def fix_torch_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

fix_torch_seed()

# Model konfigürasyonu oluştur ve yapılandır
from transformers import LlamaConfig
config = LlamaConfig()
print(config)

# Model mimarisini değiştirmek için parametreleri güncelle
config.num_hidden_layers = 12      # Katman sayısını 32'den 12'ye düşürdük
config.hidden_size = 1024          # Boyutu 4096'dan 1024'e indirgedik
config.intermediate_size = 4096    # 11008'den 4096'ya düşürüldü
config.num_key_value_heads = 8     # 32 yerine 8 olarak ayarlandı
config.torch_dtype = "bfloat16"    # Yarı hassas eğitim için ayarlandı
config.use_cache = False           # Gradients için uyumluluk sağlandı
print(config)

# Ağırlıkları rastgele başlatma
from transformers import LlamaForCausalLM
model = LlamaForCausalLM(config)
print(model)

# Toplam model parametre sayısını hesapla
def print_nparams(model):
    """Modeldeki toplam parametre sayısını hesapla."""
    nparams = sum(p.numel() for p in model.parameters())
    print(f"The total number of parameters is: {nparams}")

print_nparams(model)  # 248M parametre

# Belirli bir katmandaki ağırlıkları incele
layer_name = "model.layers.0.self_attn.q_proj.weight"

for name, param in model.named_parameters():
    if name == layer_name:
        print(f"First 30 weights of layer '{layer_name}':")
        print(param.data.view(-1)[:30])
        break

# Tokenizer'ı yükle ve test et
from transformers import LlamaTokenizer
model_dir = "./models/SOLAR-10.7B-v1.0"
tokenizer = LlamaTokenizer.from_pretrained(model_dir)

# Modelle basit bir çıkarım gerçekleştirme
from transformers import TextStreamer

prompt = "I am an engineer. I love"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

streamer = TextStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=True
)

outputs = model.generate(
    **inputs,
    streamer=streamer,
    use_cache=True,
    max_new_tokens=128,
    do_sample=False
)

# Modeli bellekten kaldır ve alanı temizle
import gc
del model
del streamer
del outputs
gc.collect()

# Önceden eğitilmiş bir modeli yeniden kullanma
from transformers import AutoModelForCausalLM

model_name_or_path = "./models/TinySolar-248m-4k"
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map="cpu",
    torch_dtype=torch.bfloat16,
)

# Belleği temizle
del model
gc.collect()

# Önceden eğitilmiş modeli katman sayısını azaltarak küçültme
from transformers import AutoTokenizer, AutoConfig

model_name_or_path = "./models/TinySolar-248m-4k"
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map="cpu",
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
print(model)
print_nparams(model)  # 248M parametre

# Orta iki katmanı çıkararak katman sayısını azalt
layers = model.model.layers
model.model.layers = layers[:5] + layers[-5:]

config = AutoConfig.from_pretrained(
    model_name_or_path,
    num_hidden_layers=len(model.model.layers),
)
model.config = config

print_nparams(model)  # 217M parametre

# Belleği temizle
del model
gc.collect()

# Önceden eğitilmiş bir modeli katman ekleyerek büyütme
config = LlamaConfig(
    num_hidden_layers=16,  # Nihai model 16 katmana sahip olacak
    hidden_size=1024,
    intermediate_size=4096,
    num_attention_heads=32,
    num_key_value_heads=8,
    torch_dtype="bfloat16",
    use_cache=False
)
print(config)

model = LlamaForCausalLM(config)
model = model.to(dtype=torch.bfloat16)  # bf16'ya dönüştür
print_nparams(model)  # 308M parametre

model_name_or_path = "upstage/TinySolar-248m-4k"
pretrained_model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map="cpu",
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

print_nparams(pretrained_model)  # 248M parametre

from copy import deepcopy

model.model.layers = deepcopy(pretrained_model.model.layers[:-4]) \
    + deepcopy(pretrained_model.model.layers[4:])

model.model.embed_tokens = deepcopy(pretrained_model.model.embed_tokens)

model.lm_head = deepcopy(pretrained_model.lm_head)

print(model.config)

# Parametre sayısının doğruluğunu kontrol et
print_nparams(model)  # 308M parametre

# Büyütülen modelle çıkarım yapmayı test et
prompt = "I am an engineer. I love"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

streamer = TextStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=True
)

outputs = model.generate(
    **inputs,
    streamer=streamer,
    use_cache=True,
    max_new_tokens=128,
    do_sample=False
)

# Modeli diske kaydet
model.save_pretrained('./data/TinySolar-308m-4k-init')
