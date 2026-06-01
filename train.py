import os
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import io

# ── Settings ──────────────────────────────────
IMAGE_SIZE = (128, 128)
EPOCHS     = 30
BATCH      = 32

# ── ELA function ──────────────────────────────
def ela(image_path):
    img = Image.open(image_path).convert('RGB')
    buf = io.BytesIO()
    img.save(buf, 'JPEG', quality=90)
    buf.seek(0)
    compressed = Image.open(buf)
    diff = ImageChops.difference(img, compressed)
    extrema = diff.getextrema()
    max_diff = max([e[1] for e in extrema]) or 1
    diff = ImageEnhance.Brightness(diff).enhance(255.0 / max_diff)
    diff = diff.resize(IMAGE_SIZE)
    return np.array(diff) / 255.0

# ── Load dataset ──────────────────────────────
# Put real images in  → dataset/real/
# Put fake images in  → dataset/fake/
X, y = [], []
for label, folder in [(0, 'dataset/real'), (1, 'dataset/fake')]:
    name = 'REAL' if label == 0 else 'FAKE'
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))]
    print(f"Loading {len(files)} {name} images...")
    for f in files:
        try:
            arr = ela(os.path.join(folder, f))
            X.append(arr)
            y.append(label)
        except:
            pass

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)
print(f"\nTotal: {len(X)} images  |  Real: {sum(y==0)}  |  Fake: {sum(y==1)}")

# ── Split ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── CNN Model ─────────────────────────────────
model = Sequential([
    Conv2D(32,  (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),

    Conv2D(64,  (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1,   activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ── Train ─────────────────────────────────────
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH, validation_split=0.1)

# ── Evaluate ──────────────────────────────────
loss, acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {acc*100:.2f}%")

y_pred = (model.predict(X_test) >= 0.5).astype(int)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Real','Fake']))

# ── Save model ────────────────────────────────
os.makedirs('model', exist_ok=True)
model.save('model/forgery_model.h5')
print("Model saved → model/forgery_model.h5")
