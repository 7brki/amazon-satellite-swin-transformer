
import gradio as gr
import torch
import timm
from torchvision import transforms
from PIL import Image
import numpy as np

device = torch.device('cpu') # Bulut sunucusunun ücretsiz CPU'sunu kullanacağız

# Modeli İnşa Et ve Ağırlıkları Yükle
model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=17)
model.load_state_dict(torch.load('swin_amazon_beyni.pth', map_location=device))
model.to(device)
model.eval()

SINIFLAR = ['agriculture', 'artisinal_mine', 'bare_ground', 'blooming', 'blow_down',
            'clear', 'cloudy', 'conventional_mine', 'cultivation', 'habitation', 
            'haze', 'partly_cloudy', 'primary', 'road', 'selective_logging', 'slash_burn', 'water']

donusum = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def uydu_analizi(image):
    img_tensor = donusum(image).unsqueeze(0).to(device)
    with torch.no_grad():
        ham_skorlar = model(img_tensor)
        olasiliklar = torch.sigmoid(ham_skorlar).numpy()[0]
    return {SINIFLAR[i]: float(olasiliklar[i]) for i in range(17)}

arayuz = gr.Interface(
    fn=uydu_analizi,
    inputs=gr.Image(type="pil", label="Uydu Görüntüsü Yükle"),
    outputs=gr.Label(num_top_classes=5, label="Yapay Zeka Analiz Sonucu"),
    title="🌍 Swin Transformer: Amazon Ormanları Uydu Radarı",
    description="2026 Standartlarında eğitilmiş Swin Transformer V1 modeli. Herhangi bir uydu fotoğrafı yükleyin ve modelin aynı anda hangi durumları (Multi-Label) tespit ettiğini görün.",
    theme="huggingface"
)

arayuz.launch()
