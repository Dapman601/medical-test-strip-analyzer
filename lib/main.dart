import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import 'inference_service.dart';

void main() {
  runApp(const StripTestApp());
}

class StripTestApp extends StatelessWidget {
  const StripTestApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Urine Strip Analyzer',
      theme: ThemeData(primarySwatch: Colors.teal),
      home: const StripHomePage(),
    );
  }
}

class StripHomePage extends StatefulWidget {
  const StripHomePage({super.key});

  @override
  State<StripHomePage> createState() => _StripHomePageState();
}

class _StripHomePageState extends State<StripHomePage> {
  final picker = ImagePicker();
  final InferenceService _service = InferenceService();

  String _result = '';
  File? _image;
  bool _isLoading = false;
  bool _isModelReady = false;

  @override
  void initState() {
    super.initState();
    _initializeInferenceService();
  }

  Future<void> _initializeInferenceService() async {
    setState(() {
      _isLoading = true;
      _result = 'Loading model...';
    });

    try {
      await _service.init();
      setState(() {
        _isModelReady = true;
        _result = 'Model loaded. Ready to analyze.';
      });
    } catch (e) {
      setState(() {
        _result = 'Failed to load model: $e';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  Future<void> _getImageAndPredict({ImageSource source = ImageSource.camera}) async {
    if (!_isModelReady) {
      setState(() {
        _result = 'Model is not ready yet.';
      });
      return;
    }

    try {
      final picked = await picker.pickImage(source: source);
      if (picked == null) return;

      final file = File(picked.path);
      final bytes = await file.readAsBytes();
      final rawImage = img.decodeImage(bytes);

      if (rawImage == null) {
        setState(() {
          _result = 'Failed to decode the image.';
        });
        return;
      }

      setState(() {
        _isLoading = true;
        _image = file;
        _result = 'Analyzing...';
      });

      // OPTION 1: Use professional validation with relaxed thresholds
      final prediction = await _service.runInference(rawImage);

      setState(() {
        _result = prediction;
      });
    } catch (e) {
      setState(() {
        _result = 'Error: $e';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Urine Strip Test')),
      body: Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: <Widget>[
              if (_isLoading) const CircularProgressIndicator(),
              if (!_isLoading)
                _image != null
                    ? Image.file(_image!, height: 200)
                    : const Text('Capture or select an image of the strip'),
              const SizedBox(height: 20),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  ElevatedButton(
                    onPressed: _isLoading ? null : () => _getImageAndPredict(source: ImageSource.camera),
                    child: const Text('Capture and Analyze'),
                  ),
                  const SizedBox(width: 16),
                  ElevatedButton(
                    onPressed: _isLoading ? null : () => _getImageAndPredict(source: ImageSource.gallery),
                    child: const Text('Select from Gallery'),
                  ),
                ],
              ),
              const SizedBox(height: 20),
              Text(
                _result,
                style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                textAlign: TextAlign.center,
              ),
            ],
          ),
        ),
      ),
    );
  }
}