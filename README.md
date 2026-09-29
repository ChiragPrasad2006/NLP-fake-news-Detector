# NLP-fake-news-Detector
Fake news detection with MiniLM-L12 for capturing clear insight over the fake news.

url: https://huggingface.co/Kami0867/MiniLM-FakeNEWS-Detection/commit/793fdc717203a8a3aeea7a40b0ceb67c60b8bb36

# Loading model:

**model_id = "Kami0867/MiniLM-FakeNEWS-Detection"**

**tokenizer = AutoTokenizer.from_pretrained(model_id)**

**model = AutoModelForSequenceClassification.from_pretrained(model_id)**


# Dataset:
dataset: https://www.kaggle.com/datasets/studymart/welfake-dataset-for-fake-news


# Achieved accuracy:
train-set : 99.869%

test-set : 99.57%

accuracy score:
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████| 201/201 [00:00<00:00, 4775.19it/s]
Map: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 64920/64920 [00:19<00:00, 3293.67 examples/s]
Map: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 7214/7214 [00:03<00:00, 2206.75 examples/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 8115/8115 [04:19<00:00, 31.23it/s]
Train accuracy: 0.9986906962415281 (0.9986906962415281)
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 902/902 [00:28<00:00, 31.84it/s]
Test accuracy: 0.9957028001108955 (0.9957028001108955)