import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
import warnings

# Игнорирование предупреждений
warnings.filterwarnings("ignore")

# Настройка параметров
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_length = 512  # Максимальная длина текста
model_path = './t5_translation_model.pth'  # Путь к сохраненной модели

# Инициализация tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=True)

# Загрузка обученной модели
config = T5Config.from_pretrained("t5-small")
model = T5ForConditionalGeneration(config)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)

# Функция для перевода текста
def translate_text(text):
    model.eval()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding='max_length'
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Пример использования
if __name__ == "__main__":
    while True:
        text_to_translate = input("Введите текст для перевода (или 'выход' для завершения): ")
        if text_to_translate.lower() == 'выход':
            break
        translated_text = translate_text(text_to_translate)
        print(f"Перевод: {translated_text}\n")