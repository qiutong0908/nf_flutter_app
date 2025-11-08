# nf_flutter_app

A Flutter-based mobile application for **on-device neurofibroma (NF) detection and classification**.  
This app performs local inference using a quantized **EfficientNet-B0** model exported from PyTorch, without requiring internet access.

---

## Features

- **On-device inference** powered by [tflite_flutter](https://pub.dev/packages/tflite_flutter)
- **Image input** via camera or gallery
- **Preprocessing** with ImageNet mean/std normalization
- **Real-time output** of label and confidence score
- **Privacy-friendly:** all processing occurs entirely on the device

---

## Folder Structure

```
lib/
├── main.dart
├── inference_service.dart # Core TFLite inference logic
├── ui/
│ ├── home_page.dart # Main interface
│ └── result_card.dart # Displays classification output
└── utils/
└── image_utils.dart # Image capture & selection helpers
assets/
└── labels.txt # NF / Non-NF / Other
models/
└── efficientnet_nf.tflite # Quantized CNN model
pubspec.yaml
```


---

## Installation & Run

```bash
flutter pub get
flutter run


