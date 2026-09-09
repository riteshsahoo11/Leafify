import requests
from PIL import Image
import io

# Create dummy image
img = Image.new('RGB', (256, 256), color = 'green')
buf = io.BytesIO()
img.save(buf, format='JPEG')
img_bytes = buf.getvalue()

try:
    res = requests.post("https://leafify-plantdiseasedetector.onrender.com/predict", files={"file": ("test.jpg", img_bytes, "image/jpeg")})
    print(res.status_code)
    print(res.text)
except Exception as e:
    print(e)
