import random
import re
import pandas as pd
import nltk
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Cargar modelos y datos
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Cargar el dataset
df = pd.read_csv('test4.csv')

# Descargar stopwords en español
nltk.download('stopwords') 
stop_words = stopwords.words('spanish') + ['hola', 'cómo', 'estás']

# Manejar valores nulos y preprocesamiento
df['Pregunta'] = df['Pregunta'].fillna('Contenido vacío')
df['Pregunta'] = df['Pregunta'].str.strip().str.lower()

# Validación de preguntas
def is_valid_question(question):
    return any(char.isalpha() for char in question)

# Filtrar preguntas válidas
df = df[df['Pregunta'].apply(is_valid_question)]

# Vectorización
vectorizer = TfidfVectorizer(stop_words=stop_words, min_df=1, max_df=0.95, ngram_range=(1, 2))
X_train, X_test, y_train, y_test = train_test_split(df['Pregunta'], df['Tag'], test_size=0.2, random_state=42)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Entrenamiento de modelo Naive Bayes
model_nb = MultinomialNB()
model_nb.fit(X_train_vec, y_train)

# Lista de saludos y respuestas
greetings = ["hola", "buenos días", "buenas tardes", "buenas noches", "hey", "saludos", "qué tal"]
greeting_responses = [
    "¡Hola! ¿En qué puedo ayudarte hoy?",
    "¡Saludos! ¿Qué necesitas saber?",
    "¡Hola! Estoy aquí para ayudarte. ¿Cuál es tu pregunta?",
    "¡Buenas! ¿Qué te gustaría preguntar?"
]

def is_spanish(text):
    pattern = r'^[a-zA-ZñÑáéíóúÁÉÍÓÚ\s,\.!?¿]+$'
    return re.match(pattern, text) is not None

def is_greeting(user_input):
    return any(greeting in user_input.lower() for greeting in greetings)

def predict_intent(user_input):
    if not is_spanish(user_input):
        return "no_entendido"
    input_encoding = tokenizer(user_input, return_tensors='pt')
    output = model(**input_encoding)
    logits = output.logits
    predicted_intent = logits.argmax().item()
    return predicted_intent

def generate_response(user_input):
    # Responder a saludos
    if is_greeting(user_input):
        return random.choice(greeting_responses)

    question_embedding = embedding_model.encode(user_input)
    dataset_embeddings = embedding_model.encode(df['Pregunta'].tolist())
    similarities = cosine_similarity([question_embedding], dataset_embeddings)[0]
    
    # Umbral de similitud
    similarity_threshold = 0.5
    most_similar_index = similarities.argmax()

    # Respuesta a preguntas no válidas
    no_response_options = [
        "Lo siento, no tengo información sobre eso. ¿Podrías reformular tu pregunta?",
        "No tengo una respuesta para eso. ¿Puedes intentar de nuevo?",
        "No puedo ayudarte con eso en este momento. ¿Podrías reformularlo?",
        "Esa pregunta no está en mi base de datos. Intenta de nuevo con otra pregunta."
    ]

    # Verificar similitud
    if similarities[most_similar_index] < similarity_threshold:
        return random.choice(no_response_options)
    
    # Obtener respuesta de la base de datos
    response = df.iloc[most_similar_index]['Respuesta']
    
    # Verificar si la respuesta es relevante usando el modelo de Naive Bayes
    user_input_vec = vectorizer.transform([user_input])
    predicted_tag = model_nb.predict(user_input_vec)[0]

    # Si el tag predicho no es relevante, retornar respuesta por defecto
    if predicted_tag not in df['Tag'].values:
        return random.choice(no_response_options)
    
    return response

@app.route('/')
def home():
    return "¡Bienvenido a la API de chat! Envía un mensaje a /chat."

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message')
    if user_input and is_valid_question(user_input):
        response = generate_response(user_input)
        return jsonify({"response": response})
    return jsonify({"response": "Por favor, ingresa un mensaje válido."})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
