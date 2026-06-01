from flask import Flask, request, render_template_string
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import tensorflow as tf
import io

app = Flask(__name__)
model = tf.keras.models.load_model('model/forgery_model.h5')

def ela(pil_img):
    img = pil_img.convert('RGB')
    buf = io.BytesIO()
    img.save(buf, 'JPEG', quality=90)
    buf.seek(0)
    compressed = Image.open(buf)
    diff = ImageChops.difference(img, compressed)
    extrema = diff.getextrema()
    max_diff = max([e[1] for e in extrema]) or 1
    diff = ImageEnhance.Brightness(diff).enhance(255.0 / max_diff)
    diff = diff.resize((128, 128))
    return np.array(diff, dtype=np.float32) / 255.0

HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Image Forgery Detection</title>
  <style>
    body { font-family: Arial, sans-serif; background: #1a1a2e; color: white; text-align: center; padding: 40px; }
    h1   { color: #e94560; }
    form { margin: 30px auto; }
    input[type=file] { padding: 10px; background: #16213e; color: white; border: 1px solid #e94560; border-radius: 6px; }
    button { padding: 12px 30px; background: #e94560; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 1rem; margin-top: 10px; }
    .result { margin: 30px auto; max-width: 400px; padding: 20px; border-radius: 10px; font-size: 1.2rem; font-weight: bold; }
    .fake { background: #c0392b; }
    .real { background: #27ae60; }
    img  { max-width: 300px; border-radius: 8px; margin-top: 15px; }
  </style>
</head>
<body>
  <h1>🔍 Image Forgery Detection</h1>
  <p>Upload any image — the system will tell you if it is Real or Fake</p>
  <form method="POST" enctype="multipart/form-data">
    <input type="file" name="image" accept="image/*" required><br>
    <button type="submit">Analyze Image</button>
  </form>

  {% if result %}
    <img src="{{ image_data }}" alt="Uploaded Image">
    <div class="result {{ 'fake' if result == 'FAKE' else 'real' }}">
      {{ '⚠️ FAKE IMAGE' if result == 'FAKE' else '✅ REAL IMAGE' }}<br>
      Confidence: {{ confidence }}%
    </div>
  {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    confidence = None
    image_data = None

    if request.method == 'POST':
        file = request.files['image']
        img = Image.open(file.stream)

        # Show uploaded image
        import base64, io as sysio
        buf = sysio.BytesIO()
        img.convert('RGB').save(buf, 'PNG')
        image_data = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

        # Predict
        ela_arr = ela(img)
        inp = np.expand_dims(ela_arr, axis=0)
        score = float(model.predict(inp, verbose=0)[0][0])

        if score >= 0.5:
            result = 'FAKE'
            confidence = round(score * 100, 2)
        else:
            result = 'REAL'
            confidence = round((1 - score) * 100, 2)

    return render_template_string(HTML, result=result, confidence=confidence, image_data=image_data)

if __name__ == '__main__':
    print("Open http://127.0.0.1:5000")
    app.run(debug=True)
