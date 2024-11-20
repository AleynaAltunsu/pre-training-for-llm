# Lesson 6. Model Evaluation

# LM Evaluation Harness kullanılarak modelin değerlendirilmesi.
# TruthfulQA MC2 görevinde TinySolar-248m-4k modelini 5 soruda değerlendirmek için kullanılan kod.

import os

# Hugging Face Leaderboard için model değerlendirme fonksiyonu.
def h6_open_llm_leaderboard(model_name):
    # Değerlendirme yapılacak görevler ve few-shot sayıları.
    task_and_shot = [
        ('arc_challenge', 25),  # Advanced Reasoning Challenge (25-shot)
        ('hellaswag', 10),      # HellaSwag (10-shot)
        ('mmlu', 5),            # Massive Multitask Language Understanding (5-shot)
        ('truthfulqa_mc2', 0),  # TruthfulQA Multiple Choice (0-shot)
        ('winogrande', 5),      # Winograd Schema Challenge (5-shot)
        ('gsm8k', 5)            # Grade School Math (5-shot)
    ]
    
    # Her görev için değerlendirme komutunun çalıştırılması.
    for task, fewshot in task_and_shot:
        eval_cmd = f"""
        lm_eval --model hf \
            --model_args pretrained={model_name} \
            --tasks {task} \
            --device cpu \
            --num_fewshot {fewshot}
        """
        os.system(eval_cmd)

# Model değerlendirmesi başlatılır.
h6_open_llm_leaderboard(model_name="YOUR_MODEL")
