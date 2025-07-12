from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        headline = request.form['headline']
        body = request.form['body']
        full_text = headline + " " + body

        vec = vectorizer.transform([full_text])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0][1]

        result = "🟢 REAL" if pred == 1 else "🔴 FAKE"
        confidence = f"{proba*100:.2f}%"

        return render_template("index.html", result=result, confidence=confidence,
                               headline=headline, body=body)

    return render_template("index.html", result=None)
    
if __name__ == '__main__':
    app.run(debug=True)
