import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import copy
from main import adaptive_healing
from train import CNN, get_train_loader
from fault import inject_weight_fault, detect_low_gradient_weights
from heal import neuron_regrowth, micro_finetune_fc

IMAGE_PATH = "test_images\download (2).webp"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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



def predict(model, image_tensor):

    model.eval()

    with torch.no_grad():

        output = model(image_tensor)

        probs = torch.softmax(output, dim=1)

        conf, pred = torch.max(probs, 1)

    return classes[pred.item()], conf.item()*100


def load_model():

    model = CNN().to(device)

    model.load_state_dict(
        torch.load("models/baseline_cnn.pth", map_location=device)
    )

    model.eval()

    return model


def main():

    print(f"Loading image: {IMAGE_PATH}")

    image = Image.open(IMAGE_PATH).convert("RGB")

    image_tensor = transform(image).unsqueeze(0).to(device)


    model = load_model()

    label, conf = predict(model, image_tensor)

    print("\nBaseline Prediction")
    print(f"Class : {label}")
    print(f"Confidence : {conf:.2f}%")


    faulty_model = copy.deepcopy(model)

    faulty_model = inject_weight_fault(faulty_model, fault_ratio=0.2)

    label, conf = predict(faulty_model, image_tensor)

    print("\nAfter Fault Injection")
    print(f"Class : {label}")
    print(f"Confidence : {conf:.2f}%")

    print("\nRunning Healing...")

    train_loader = get_train_loader()

    grad_faults, grad_vals = detect_low_gradient_weights(
        faulty_model,
        train_loader,
        nn.CrossEntropyLoss(),
        device,
        max_batches=2
    )

    fault_ratio = 0.2
    healed_model = adaptive_healing(
        faulty_model,
        fault_ratio,
        grad_faults,
        grad_vals
    )

    healed_model = copy.deepcopy(healed_model)

    healed_model = micro_finetune_fc(
        healed_model,
        train_loader,
        device,
        epochs=2
    )

    label, conf = predict(healed_model, image_tensor)

    print("\nAfter Healing")
    print(f"Class : {label}")
    print(f"Confidence : {conf:.2f}%")


if __name__ == "__main__":
    main()