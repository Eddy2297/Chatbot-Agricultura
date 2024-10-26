# Chatbot de Agricultura - Departamento de Antioquia

Este proyecto es un chatbot desarrollado para proporcionar respuestas a preguntas relacionadas con la agricultura, el mantenimiento y el cuidado de diferentes cultivos en el departamento de Antioquia, como banano, lechuga, aguacate, plátano, papa, entre otros. Fue inicialmente desarrollado en Google Collaboratory y luego desplegado en una aplicación Flask utilizando Visual Studio.

## Descripción del Proyecto

El chatbot utiliza un modelo de procesamiento de lenguaje natural (NLP) basado en BERT para comprender las preguntas de los usuarios y proporcionar respuestas relevantes. El dataset contiene preguntas y respuestas sobre agricultura en Antioquia y está diseñado para ayudar a los agricultores a obtener información útil sobre los cultivos.

### Características:

- Responde a preguntas sobre diferentes cultivos agrícolas.
- Responde a saludos de manera amigable.
- Utiliza técnicas de vectorización y clasificación de texto para identificar preguntas válidas y relevantes.
- Integra un modelo Naive Bayes para verificar la relevancia de las respuestas.

## Requisitos

- Python 3.7 o superior
- Flask 2.0 o superior
- pandas
- transformers
- sentence-transformers
- scikit-learn
- nltk
- flask-cors

### Instalación de Dependencias

Para instalar todas las dependencias necesarias, puedes ejecutar:

```bash
pip install pandas flask transformers sentence-transformers scikit-learn nltk flask-cors


Dataset
El dataset test4.csv contiene las siguientes columnas:

Pregunta: La pregunta relacionada con un cultivo o práctica agrícola.
Respuesta: La respuesta correspondiente para esa pregunta.
Tag: Etiquetas que categorizan las preguntas por tipo de cultivo o tema agrícola.
Estructura del Código
app.py: Contiene la lógica principal del chatbot.
modelos: Se carga un modelo de clasificación basado en BERT para la comprensión del lenguaje y un modelo Naive Bayes para la validación de relevancia.
vectorización: El texto de las preguntas es vectorizado utilizando TF-IDF, eliminando palabras irrelevantes para mejorar la precisión de las respuestas.
Embeddings: Las preguntas y las respuestas se comparan utilizando embeddings generados con Sentence Transformers y se calculan similitudes para identificar la mejor respuesta.
Lógica del Chatbot
Saludo: El chatbot responde de manera amigable si detecta un saludo en el mensaje del usuario.
Clasificación de Preguntas: Si el input es una pregunta, se calcula la similitud con las preguntas del dataset utilizando cosine_similarity.
Respuesta Predeterminada: Si no hay una pregunta válida o no se encuentra una respuesta relevante, el chatbot proporciona una respuesta predeterminada pidiendo más información.
Ejecución del Proyecto
Clona este repositorio.
Asegúrate de que todas las dependencias están instaladas.
Ejecuta la aplicación Flask desde Visual Studio o cualquier entorno local.

python app.py

El chatbot estará disponible en http://localhost:5001/chat y recibirá solicitudes POST con el formato:

{
    "message": "¿Cómo puedo cuidar mis plantas de banano?"
}


Despliegue en Producción
Para desplegar este chatbot en un servidor de producción, sigue estos pasos:

Configura un servidor (por ejemplo, AWS, Heroku, etc.).
Asegúrate de que las dependencias estén instaladas.
Configura un puerto de producción y cambia el modo debug a False en el archivo principal de Flask.
Utiliza gunicorn o uwsgi para gestionar el despliegue de Flask en producción.
Colaboradores
Este proyecto fue desarrollado por un equipo de estudiantes que hicieron parte del bootcamp de mintic sobre inteligencia artificial y talento tech version 2. 

Estudiantes
Laura Sofia Luna Duque
Edilberto Salazar Garcia
Christian Javier Uchima Sierra
Adriana Maria Velez
Ismael Vasco



