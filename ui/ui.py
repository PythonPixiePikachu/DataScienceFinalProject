from flask import Flask, request, render_template, redirect, send_from_directory
from werkzeug.utils import secure_filename
import os
import requests
from PIL import Image
from io import BytesIO
from transformers import AutoProcessor
from transformers import AutoModelForCausalLM


checkpoint = "microsoft/git-base"
processor = AutoProcessor.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(r"C:\Users\User\ui\checkpoint-910")


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

PEXELS_API_KEY = 'fikXUtk7NuaDJaBJbsZKwhPqAYY5WwqxRDG3gshLz84MtIMNqrGQcZhy'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            caption = predict_caption(filepath)
            return render_template('index.html', filename=filename, caption=caption)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/pexels', methods=['GET', 'POST'])
def pexels():
    query = request.args.get('query')
    photos = []
    if query:
        headers = {
            'Authorization': PEXELS_API_KEY
        }
        response = requests.get(f'https://api.pexels.com/v1/search?query={query}&per_page=10', headers=headers)
        if response.status_code == 200:
            data = response.json()
            photos = data.get('photos', [])
    return render_template('pexels.html', photos=photos)

@app.route('/pexels/predict', methods=['POST'])
def pexels_predict():
    image_url = request.form['image_url']
    caption = predict_caption_from_url(image_url)
    return render_template('pexels.html', caption=caption, image_url=image_url, photos=[])

def predict_caption(filepath):
    image = Image.open(filepath)
    inputs = processor(images=image, return_tensors="pt").to('cpu')
    pixel_values = inputs.pixel_values
    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_caption
    

def predict_caption_from_url(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    inputs = processor(images=img, return_tensors="pt").to('cpu')
    pixel_values = inputs.pixel_values
    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_caption


if __name__ == '__main__':
    app.run(debug=True)
    

