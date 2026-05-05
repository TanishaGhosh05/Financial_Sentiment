from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

app = Flask(__name__)

# Load FinBERT model
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
financial_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        text = request.form['text']
        analysis = financial_pipeline(text)[0]
        sentiment_label = analysis['label'].capitalize()
        score = round(analysis['score'] * 100, 2)
        result = f"{sentiment_label} sentiment with confidence {score}%"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
