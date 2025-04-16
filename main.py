from flask import Flask, request, jsonify,render_template
import google.generativeai as genai
from nltk.tokenize import sent_tokenize, word_tokenize
import re
from nltk import FreqDist
from nltk.corpus import stopwords
import nltk

app = Flask(__name__)

# Download NLTK data (run this once)
nltk.download('punkt')
nltk.download('stopwords')


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


def convert_paragraph_to_points(paragraph, num_points=5):
    sentences = sent_tokenize(paragraph)
    words = word_tokenize(paragraph.lower())
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    freq_dist = FreqDist(filtered_words)
    sentence_scores = {}
    for sentence in sentences:
        sentence_word_tokens = word_tokenize(sentence.lower())
        sentence_word_tokens = [word for word in sentence_word_tokens if word.isalnum()]
        score = sum(freq_dist.get(word, 0) for word in sentence_word_tokens)
        sentence_scores[sentence] = score
    sorted_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    key_points = sorted_sentences[:num_points]
    return key_points


def clean_text(text):
    return re.sub(r'\*\*|\*', '', text)


@app.route('/chat', methods=['POST'])
def chatting():
    if request.method == 'POST':
        try:
            data = request.get_json()
            user_message = data.get('message', '')
            context = data.get('context', 'general')

            genai.configure(api_key='AIzaSyCOoAQyClkN6jGPl5iskpU0knbnERA-gVE')
            model = genai.GenerativeModel('gemini-1.5-flash')

            # Context-based prompts
            if context == '1':  # FAQ
                prompt = f"Provide a concise answer to this FAQ question: {user_message}"
            elif context == '2':  # Account
                prompt = f"Help with account-related query: {user_message}"
            elif context == '3':  # Products
                prompt = f"Provide information about products related to: {user_message}"
            elif context == '4':  # Support
                prompt = f"Help with support issue: {user_message}"
            elif context == '5':  # About
                prompt = f"Provide information about our company related to: {user_message}"
            else:
                prompt = f"Respond to this general inquiry: {user_message}"

            response = model.generate_content(prompt)
            generated_text = response.text
            if context in ['1', '3', '4']:  # FAQ, Products, Support
                key_points = convert_paragraph_to_points(generated_text)
                key_points = [clean_text(item) for item in key_points]
                formatted_response = "Here are the key points:\n- " + "\n- ".join(key_points)
            else:
                formatted_response = clean_text(generated_text)
            return jsonify({'response': formatted_response})
        except Exception as e:
            print(f"Error: {str(e)}")
            return jsonify({'response': "Sorry, I encountered an error processing your request."}), 500


if __name__ == '__main__':
    app.run(debug=True)