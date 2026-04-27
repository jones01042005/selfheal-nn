import os
import copy
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import torchvision.transforms as transforms
import threading
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Global definitions to ensure one-time strict execution
from train import CNN, get_val_loader, get_train_loader
from main import adaptive_healing
from fault import inject_weight_fault, detect_low_gradient_weights
from heal import micro_finetune_fc

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using target device:", device)

classes = [
    "Airplane","Automobile","Bird","Cat","Deer",
    "Dog","Frog","Horse","Ship","Truck"
]

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914,0.4822,0.4465),
        (0.2023,0.1994,0.2010)
    )
])

print("Startup Event: Pulling models into persistent VRAM.")
cnn_model = CNN().to(device)

try:
    cnn_model.load_state_dict(torch.load("models/baseline_cnn.pth", map_location=device, weights_only=True))
except TypeError:
    cnn_model.load_state_dict(torch.load("models/baseline_cnn.pth", map_location=device))

cnn_model.eval()

GLOBAL_MODELS = {
    'baseline': cnn_model,
    'faulty': None,
    'healed': None
}

print("Startup Event: Injecting Datasets.")
val_loader = get_val_loader()
train_loader = get_train_loader()


def predict(model, image_tensor):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
    return classes[pred.item()], conf.item()*100

def fast_evaluate(model, loader, max_batches=2):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= max_batches: break
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total

def save_healing_plot(base_acc, fault_acc, healed_acc, fault_ratio):
    plt.figure(figsize=(6, 4))
    categories = ['Baseline', 'Faulty', 'Healed']
    accuracies = [base_acc, fault_acc, healed_acc]
    colors = ['#3b82f6', '#ef4444', '#10b981']
    plt.bar(categories, accuracies, color=colors)
    plt.ylabel('Accuracy (%)')
    plt.title(f'Healing Comparison (Fault Ratio: {fault_ratio*100:.0f}%)')
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    os.makedirs('experiments', exist_ok=True)
    plot_path = os.path.join('experiments', 'ui_healing_results.png')
    plt.savefig(plot_path)
    plt.close()
    
cached_base_acc = None

@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/api/baseline', methods=['POST'])
def run_baseline():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    file = request.files['image']
    
    try:
        image = Image.open(file.stream).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        return jsonify({'error': 'Invalid image format'}), 400

    global cached_base_acc
    if cached_base_acc is None:
        GLOBAL_MODELS['baseline'].eval()
        cached_base_acc = fast_evaluate(GLOBAL_MODELS['baseline'], val_loader, max_batches=2)

    label, conf = predict(GLOBAL_MODELS['baseline'], image_tensor)

    return jsonify({
        'prediction': label,
        'confidence': f"{conf:.2f}",
        'accuracy': f"{cached_base_acc:.2f}"
    })


@app.route('/api/fault', methods=['POST'])
def run_fault():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    file = request.files['image']
    fault_ratio_str = request.form.get('fault_ratio', '20')
    fault_ratio = float(fault_ratio_str) / 100.0
    
    try:
        image = Image.open(file.stream).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        return jsonify({'error': 'Invalid image format'}), 400

    # Execute Safe Deep Copy 1
    GLOBAL_MODELS['faulty'] = copy.deepcopy(GLOBAL_MODELS['baseline'])
    GLOBAL_MODELS['faulty'] = inject_weight_fault(GLOBAL_MODELS['faulty'], fault_ratio=fault_ratio)

    f_label, f_conf = predict(GLOBAL_MODELS['faulty'], image_tensor)
    
    GLOBAL_MODELS['faulty'].eval()
    fault_acc = fast_evaluate(GLOBAL_MODELS['faulty'], val_loader, max_batches=2)

    return jsonify({
        'prediction': f_label,
        'confidence': f"{f_conf:.2f}",
        'accuracy': f"{fault_acc:.2f}"
    })


@app.route('/api/heal', methods=['POST'])
def run_heal():
    if 'image' not in request.files:
         return jsonify({'error': 'No image provided'}), 400
    if GLOBAL_MODELS['faulty'] is None:
         return jsonify({'error': 'Fault model not injected yet'}), 400
         
    file = request.files['image']
    fault_ratio_str = request.form.get('fault_ratio', '20')
    fault_ratio = float(fault_ratio_str) / 100.0
    
    try:
        image = Image.open(file.stream).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        return jsonify({'error': 'Invalid image format'}), 400


    # Execute Safe Deep Copy 2
    GLOBAL_MODELS['healed'] = copy.deepcopy(GLOBAL_MODELS['faulty'])

    # Strict Limits Execution
    GLOBAL_MODELS['healed'].train()
    grad_faults, grad_vals = detect_low_gradient_weights(
        GLOBAL_MODELS['healed'], train_loader, nn.CrossEntropyLoss(), device, max_batches=1
    )
    
    GLOBAL_MODELS['healed'] = adaptive_healing(GLOBAL_MODELS['healed'], fault_ratio, grad_faults, grad_vals)
    
    GLOBAL_MODELS['healed'].train()
    GLOBAL_MODELS['healed'] = micro_finetune_fc(GLOBAL_MODELS['healed'], train_loader, device, epochs=3, max_batches=10)
    

    h_label, h_conf = predict(GLOBAL_MODELS['healed'], image_tensor)
    
    GLOBAL_MODELS['healed'].eval()
    healed_acc = fast_evaluate(GLOBAL_MODELS['healed'], val_loader, max_batches=2)
    
    # Calculate recovery explicitly
    GLOBAL_MODELS['faulty'].eval()
    fault_acc = fast_evaluate(GLOBAL_MODELS['faulty'], val_loader, max_batches=2)
    
    global cached_base_acc
    if cached_base_acc != fault_acc:
        recovery = ((healed_acc - fault_acc) / (cached_base_acc - fault_acc)) * 100
    else:
        recovery = 0.0

    threading.Thread(
        target=save_healing_plot, 
        args=(cached_base_acc, fault_acc, healed_acc, fault_ratio)
    ).start()

    return jsonify({
        'prediction': h_label,
        'confidence': f"{h_conf:.2f}",
        'accuracy': f"{healed_acc:.2f}",
        'fault_acc': f"{fault_acc:.2f}", # added for safety in chart building
        'recovery': f"{recovery:.2f}"
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)
