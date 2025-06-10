Чтобы установить все необходимые зависимости для серверной части и обучения модели, добавьте следующие инструкции в ваш Markdown-файл. Я разделил установку для серверной части и обучения, а также добавил варианты для PyTorch с поддержкой GPU и CPU.


#### Ссылка на скачивание модели - https://disk.yandex.ru/d/twRLLu2DtTR1hw
После скачивания загрузить по пути `...\Diplom\API\data`

---

#### Ссылка на скачивание набора для обучения - https://disk.yandex.ru/d/W-Ip4-F6nJyEow
После скачивания загрузить по пути `...\Diplom\neero\data`

---

## Установка зависимостей

### Для серверной части (API):

#### Основные зависимости
```pip install fastapi uvicorn sqlalchemy python-docx pdfminer.six PyMuPDF reportlab googletrans==4.0.0-rc1```

#### PyTorch для CPU (если не используется GPU)
```pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu```

#### ИЛИ PyTorch с поддержкой CUDA (для GPU NVIDIA)
```pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118```

### Для обучения модели:

### Основные зависимости
```pip install pandas scikit-learn matplotlib nltk tensorboard tqdm rich```

### Дополнительно для NLTK
```python -c "import nltk; nltk.download('punkt')"```

## Установите PyTorch (выберите нужную версию из вышеприведенных команд)

### Важные примечания:
1. **Выбор версии PyTorch**:
   - Для систем **без GPU** используйте CPU-версию
   - Для систем **с NVIDIA GPU** используйте CUDA 11.8-версию
   - Проверьте наличие драйверов CUDA: `nvidia-smi` (должна быть версия 11.8+)

2. **Требования к системе**:
   - Для обучения рекомендуется GPU с 8+ GB памяти
   - Для серверной части достаточно CPU (но GPU ускорит работу)

3. **Проблемы с googletrans**:
   - Используется версия 4.0.0-rc1 для избежания ошибок
   - При проблемах попробуйте:```pip uninstall googletrans```, затем ```pip install googletrans==4.0.0-rc1```

4. **Версии библиотек**:
   - Все команды тестировались с Python 3.8-3.10
   - Рекомендуется использовать виртуальное окружение


### Пояснения к установке:
1. **PyTorch с GPU/CPU поддержкой**:
   - Официальные сборки PyTorch имеют разные версии для CPU и GPU
   - CUDA-версия требует наличия видеокарты NVIDIA и установленных драйверов
   - CPU-версия работает на любом оборудовании

2. **Особые случаи**:
   - **Google Translate (googletrans)**: Версия 4.0.0-rc1 стабильнее последних релизов
   - **NLTK Data**: Требуется скачивание дополнительных данных (`punkt` для токенизации)
   - **PDF обработка**: PyMuPDF (fitz) требует системных зависимостей (libgl1 на Linux)

3. **Проверка установки PyTorch**:

import torch
```print(f"PyTorch version: {torch.__version__}")```
```print(f"CUDA available: {torch.cuda.is_available()}")```
```print(f"CUDA version: {torch.version.cuda}")```


4. **Для Linux-систем** может потребоваться установка:

# Для PyMuPDF (fitz)
```sudo apt install libgl1```
