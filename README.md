This project classifies tweets as disaster-related or not using BERT (bert-large-uncased). The model is fine-tuned on the "Real or Not? NLP with Disaster Tweets" dataset from Kaggle.

Steps
Preprocess Data: Lowercase the text, remove special characters, emails, and HTML tags.
BERT Tokenization: Tokenize the tweets with a max sequence length of 36 tokens, generating input IDs and attention masks.
Model: Fine-tune the pre-trained bert-large-uncased model using a custom Keras architecture with additional dense layers for binary classification.
Training: The model is trained for 10 epochs with binary cross-entropy loss and Adam optimizer.
Prediction: The fine-tuned model predicts whether tweets are disaster-related or not, and results are formatted for submission.
