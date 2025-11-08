import 'dart:io';
import 'dart:async';
import 'dart:typed_data';
import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart';
import 'package:flutter_pytorch_lite/flutter_pytorch_lite.dart' as fpl;

class NFScreen extends StatefulWidget {
  const NFScreen({super.key});
  @override
  State<NFScreen> createState() => _NFScreenState();
}

class _NFScreenState extends State<NFScreen> {
  fpl.Module? _module; // Loaded native module
  List<String> _labels = [];
  File? _img;
  String? _pred;
  double? _conf;
  String? _analysis; // Analysis text

  @override
  void initState() {
    super.initState();
    _load();
  }

  /// Copy model to a writable path (required on iOS) and load it.
  Future<void> _load() async {
    try {
      debugPrint('[NF] _load start');

      // 1) Load labels
      final labelsStr = await rootBundle.loadString('assets/models/labels.txt');
      _labels = labelsStr.trim().split('\n');
      debugPrint('[NF] labels: $_labels');

      // 2) Copy model to a temporary dir
      final data = await rootBundle.load(
        'assets/models/nf_cls_ts.ptl',
      ); // ByteData
      final dir = await getTemporaryDirectory();
      final modelPath = '${dir.path}/nf_cls_ts.ptl';
      await File(
        modelPath,
      ).writeAsBytes(data.buffer.asUint8List(), flush: true);
      debugPrint(
        '[NF] model copied: $modelPath size=${await File(modelPath).length()}',
      );

      // 3) Load model (native)
      _module = await fpl.FlutterPytorchLite.load(modelPath);
      debugPrint('[NF] model loaded OK');
    } catch (e, st) {
      debugPrint('[NF] _load error: $e\n$st');
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Failed to load model: $e')));
      }
      _module = null;
    } finally {
      if (mounted) setState(() {});
    }
  }

  /// Pick image -> 320x320 -> normalize (NCHW) -> forward -> softmax -> analysis
  Future<void> _pickAndPredict([ImageSource src = ImageSource.gallery]) async {
    final x = await ImagePicker().pickImage(source: src);
    if (x == null) return;

    _img = File(x.path);
    setState(() {
      _pred = null;
      _conf = null;
      _analysis = null;
    });

    if (_module == null) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text('Model is not loaded yet')));
      return;
    }

    // A) Load as ui.Image and resize to 320x320
    final uiImage = await _fileToUiImage(_img!);
    final ui.Image resized = fpl.TensorImageUtils.resizeImage(
      uiImage,
      320,
      320,
    );

    // B) Extract RGBA float32 and normalize to NCHW
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    final byteData = await resized.toByteData(
      format: ui.ImageByteFormat.rawExtendedRgba128,
    );
    final rgba =
        byteData!.buffer
            .asFloat32List(); // per pixel 4 floats (R,G,B,A) in 0..1
    const int H = 320, W = 320, C = 3;
    final chw = Float32List(H * W * C); // [C,H,W]

    final int pixels = H * W;
    for (int i = 0; i < pixels; i++) {
      final r = rgba[i * 4 + 0];
      final g = rgba[i * 4 + 1];
      final b = rgba[i * 4 + 2];
      chw[i + 0 * pixels] = (r - mean[0]) / std[0];
      chw[i + 1 * pixels] = (g - mean[1]) / std[1];
      chw[i + 2 * pixels] = (b - mean[2]) / std[2];
    }

    final shape = Int64List.fromList([1, C, H, W]);
    final tensor = fpl.Tensor.fromBlobFloat32(chw, shape);
    final input = fpl.IValue.from(tensor); // Your version exposes IValue.from

    // C) Forward and parse logits
    final outIValue = await _module!.forward([input]); // IValue
    final t = outIValue.toTensor(); // abstract Tensor

    late List<double> flat; // flattened logits
    if (t is fpl.TensorFloat32) {
      flat = t.data.toList().map((e) => e.toDouble()).toList();
    } else {
      final dynamic raw = (t as dynamic).data;
      if (raw is Float32List) {
        flat = raw.toList().map((e) => e.toDouble()).toList();
      } else if (raw is List) {
        flat = raw.cast<num>().map((e) => e.toDouble()).toList();
      } else {
        throw StateError('Unexpected tensor data type: ${raw.runtimeType}');
      }
    }

    // Support [num_classes] or [1, num_classes]
    final int nc = _labels.length;
    final List<double> logits =
        (flat.length == nc) ? flat : flat.sublist(0, nc);

    // Softmax
    final exps = logits.map((v) => math.exp(v)).toList();
    final sum = exps.fold<double>(0.0, (a, b) => a + b);
    final probs = exps.map((e) => e / sum).toList();

    final idx = _argmax(probs);
    final label = _labels[idx];
    final prob = probs[idx];

    setState(() {
      _pred = label;
      _conf = prob;
      _analysis = _makeAnalysis(label, prob);
    });
  }

  Future<ui.Image> _fileToUiImage(File f) async {
    final bytes = await f.readAsBytes();
    final c = Completer<ui.Image>();
    ui.decodeImageFromList(bytes, (img) => c.complete(img));
    return c.future;
  }

  int _argmax(List<double> xs) {
    var best = 0;
    var v = xs[0];
    for (var i = 1; i < xs.length; i++) {
      if (xs[i] > v) {
        v = xs[i];
        best = i;
      }
    }
    return best;
  }

  // ===== Analysis text =====
  String _makeAnalysis(String label, double p) {
    // thresholds can be tuned later
    const hi = 0.85, mid = 0.60;
    final l = label.toLowerCase(); // 'nf' | 'nonnf' | 'other'

    if (l == 'other') {
      return 'Predicted as “other” (not a skin/lesion photo).\n'
          'Recommendations:\n'
          '• Please upload a clear skin/lesion photo (centered, in focus, even lighting).\n'
          '• Avoid documents, landscapes, objects, text, or screenshots.\n'
          '• If you have clinical concerns, consider in-person evaluation.';
    }

    if (l == 'nf') {
      if (p >= hi) {
        return 'The model strongly suggests Neurofibroma (NF).\n'
            'Recommendations:\n'
            '• If new or changing, seek clinical evaluation; dermoscopy or pathology when appropriate.\n'
            '• Capture multiple clear angles for follow-up.';
      } else if (p >= mid) {
        return 'Leaning NF with moderate confidence.\n'
            'Recommendations:\n'
            '• Monitor size/color/shape over 2–4 weeks.\n'
            '• If pain, itching, bleeding, or rapid growth occurs, seek care promptly.';
      } else {
        return 'Weak positive — insufficient evidence.\n'
            'Recommendations:\n'
            '• Re-take a well-lit, in-focus close-up and try again.\n'
            '• Consider in-person evaluation if symptoms are concerning.';
      }
    }

    // nonNF
    if (p >= hi) {
      return 'Likely a non-NF benign lesion.\n'
          'Recommendations:\n'
          '• Observation is reasonable if asymptomatic; keep periodic photos.\n'
          '• If rapid growth, asymmetry, irregular border, or color variegation appears, seek care.';
    } else if (p >= mid) {
      return 'Leaning non-NF with moderate confidence.\n'
          'Recommendations:\n'
          '• Continue observation; consider clinic visit if changes persist.';
    } else {
      return 'Model uncertainty is high.\n'
          'Recommendations:\n'
          '• Re-take a clearer photo and re-check; visit clinic if needed.';
    }
  }

  // ===== UI =====
  @override
  Widget build(BuildContext context) {
    final ready = _module != null;
    final hasImage = _img != null;

    return Scaffold(
      appBar: AppBar(
        title: const Text('NF Detector (Research)'),
        centerTitle: true,
      ),
      body: SafeArea(
        child: Column(
          children: [
            _statusCard(ready),
            const SizedBox(height: 12),

            Expanded(
              child: AnimatedSwitcher(
                duration: const Duration(milliseconds: 250),
                child:
                    hasImage
                        ? SingleChildScrollView(
                          key: const ValueKey('withImage'),
                          padding: const EdgeInsets.symmetric(horizontal: 16),
                          child: Column(
                            children: [
                              ClipRRect(
                                borderRadius: BorderRadius.circular(16),
                                child: Image.file(_img!, fit: BoxFit.cover),
                              ),
                              const SizedBox(height: 12),
                              if (_pred != null) ...[
                                Row(
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  children: [
                                    Chip(
                                      label: Text(
                                        'Prediction: $_pred',
                                        style: const TextStyle(
                                          fontWeight: FontWeight.w600,
                                        ),
                                      ),
                                    ),
                                  ],
                                ),
                                const SizedBox(height: 8),
                                _confidenceBar(_conf ?? 0),
                                if (_analysis != null) ...[
                                  const SizedBox(height: 10),
                                  _analysisCard(_analysis!),
                                ],
                              ],
                            ],
                          ),
                        )
                        : _placeholder(ready),
              ),
            ),

            Padding(
              padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
              child: Row(
                children: [
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed:
                          ready
                              ? () => _pickAndPredict(ImageSource.gallery)
                              : null,
                      icon: const Icon(Icons.photo),
                      label: const Text('Pick from Gallery'),
                      style: ElevatedButton.styleFrom(
                        minimumSize: const Size.fromHeight(48),
                      ),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed:
                          ready
                              ? () => _pickAndPredict(ImageSource.camera)
                              : null,
                      icon: const Icon(Icons.camera_alt),
                      label: const Text('Take Photo'),
                      style: OutlinedButton.styleFrom(
                        minimumSize: const Size.fromHeight(48),
                      ),
                    ),
                  ),
                ],
              ),
            ),

            const Padding(
              padding: EdgeInsets.only(bottom: 8),
              child: Text(
                'For research use only — not a medical diagnosis.',
                style: TextStyle(fontSize: 12),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _statusCard(bool ready) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Container(
        decoration: BoxDecoration(
          color: ready ? const Color(0xFFE8F5E9) : const Color(0xFFFFF3E0),
          borderRadius: BorderRadius.circular(12),
        ),
        padding: const EdgeInsets.all(12),
        child: Row(
          children: [
            Icon(
              ready ? Icons.check_circle : Icons.hourglass_top,
              color: ready ? Colors.green : Colors.orange,
            ),
            const SizedBox(width: 8),
            Expanded(
              child: Text(
                ready
                    ? 'Model loaded — ready to detect'
                    : 'Loading model… the first run may take a few seconds',
                style: const TextStyle(fontSize: 14),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _placeholder(bool ready) {
    return Center(
      key: const ValueKey('placeholder'),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 24),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.image_search, size: 96, color: Colors.black26),
            const SizedBox(height: 12),
            const Text(
              'Upload or take a photo of a skin lesion',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
            ),
            const SizedBox(height: 8),
            const Text(
              'Tips: center the lesion, ensure focus and even lighting, avoid glare.',
              textAlign: TextAlign.center,
              style: TextStyle(color: Colors.black54),
            ),
            const SizedBox(height: 16),
            ElevatedButton.icon(
              onPressed:
                  ready ? () => _pickAndPredict(ImageSource.gallery) : null,
              icon: const Icon(Icons.upload),
              label: const Text('Pick from Gallery'),
              style: ElevatedButton.styleFrom(minimumSize: const Size(240, 44)),
            ),
            const SizedBox(height: 8),
            TextButton.icon(
              onPressed:
                  ready ? () => _pickAndPredict(ImageSource.camera) : null,
              icon: const Icon(Icons.camera_alt),
              label: const Text('Take Photo'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _confidenceBar(double p) {
    return Column(
      children: [
        Text(
          'Confidence: ${p.toStringAsFixed(3)}',
          style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
        ),
        const SizedBox(height: 6),
        ClipRRect(
          borderRadius: BorderRadius.circular(8),
          child: LinearProgressIndicator(value: p.clamp(0, 1), minHeight: 10),
        ),
      ],
    );
  }

  Widget _analysisCard(String text) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFFEEF2FF),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.insights, color: Color(0xFF3F51B5)),
          const SizedBox(width: 8),
          Expanded(child: Text(text, style: const TextStyle(height: 1.4))),
        ],
      ),
    );
  }
}
