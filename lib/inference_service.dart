import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'disease_labels.dart';

class InferenceService {
  late Interpreter interpreter;
  late DiseaseLabels labels;
  final int inputSize = 512;
  bool _isInitialized = false;
  late int numClasses;

  // Valid medical parameter names
  static const Set<String> validMedicalParameters = {
    'Glucose', 'Bilirubin', 'Ketones', 'Specific Gravity', 'Blood',
    'pH', 'Protein', 'Urobilinogen', 'Nitrite', 'Leukocytes'
  };

  void _log(String message) {
    if (kDebugMode) {
      debugPrint(message);
    }
  }

  Future<void> init() async {
    try {
      interpreter = await Interpreter.fromAsset('assets/model_with_metadata.tflite');

      // ✅ CRITICAL: Allocate tensors before use
      interpreter.allocateTensors();

      // ✅ Get actual output shape from model
      final outputTensor = interpreter.getOutputTensor(0);
      numClasses = outputTensor.shape.last; // Usually [1, numClasses]
      _log("Model input shape: ${interpreter.getInputTensor(0).shape}");
      _log("Model output shape: ${outputTensor.shape}");
      _log("Number of classes: $numClasses");

      labels = DiseaseLabels();
      await labels.loadLabels('assets/annotations_fixed.coco.json');

      // Check which labels are medical parameters
      final allLabels = labels.getAllLabels();
      final medicalLabels = allLabels.where((label) => _isMedicalParameter(label)).toList();
      _log("All labels: ${allLabels.join(', ')}");
      _log("Medical parameter labels: ${medicalLabels.join(', ')}");

      _isInitialized = true;
      _log("✅ Model and labels loaded successfully.");
    } catch (e) {
      _log("❌ Error initializing inference service: $e");
      rethrow;
    }
  }

  static bool _isMedicalParameter(String label) {
    // Filter out numerical disease labels, keep only medical test parameters
    final isValid = validMedicalParameters.contains(label.trim());
    final isNumericOrDisease = RegExp(r'^\d+$').hasMatch(label.trim()) ||
        label.toLowerCase().contains('disease') ||
        label.toLowerCase().contains('condition') ||
        label.toLowerCase().contains('class');

    return isValid && !isNumericOrDisease;
  }

  Future<String> runInference(img.Image originalImage) async {
    if (!_isInitialized) {
      throw StateError("❌ Interpreter not initialized. Call init() before inference.");
    }

    try {
      // Basic quality check
      final qualityWarnings = _checkBasicQuality(originalImage);

      final img.Image processedImage = processImage(originalImage);
      final inputBuffer = imageToBuffer(processedImage);

      // ✅ Create proper 4D input tensor shape [batch, height, width, channels]
      final input = List.generate(1, (_) =>
          List.generate(inputSize, (_) =>
              List.generate(inputSize, (_) =>
                  List.generate(3, (_) => 0.0)
              )
          )
      );

      // Fill the input tensor
      int bufferIndex = 0;
      for (int b = 0; b < 1; b++) {
        for (int y = 0; y < inputSize; y++) {
          for (int x = 0; x < inputSize; x++) {
            for (int c = 0; c < 3; c++) {
              input[b][y][x][c] = inputBuffer[bufferIndex++];
            }
          }
        }
      }

      // ✅ Use dynamic output size based on model
      var output = List.generate(1, (_) => List.filled(numClasses, 0.0));

      interpreter.run(input, output);

      // Find all medical parameters with their confidences
      final medicalResults = <String, double>{};
      for (int i = 0; i < output[0].length; i++) {
        final label = labels.getLabel(i);
        final confidence = output[0][i];

        // Only include medical parameters, skip disease classifications
        if (_isMedicalParameter(label)) {
          medicalResults[label] = confidence;
          _log("Medical parameter: $label ($i) -> ${(confidence * 100).toStringAsFixed(2)}%");
        } else if (label != "Unknown") {
          _log("🚫 Skipping non-medical label: $label -> ${(confidence * 100).toStringAsFixed(2)}%");
        }
      }

      // Format results for medical parameters only
      if (medicalResults.isEmpty) {
        return "⚠️ No medical parameters detected.\nOnly test strip parameters (Glucose, Protein, etc.) are analyzed.";
      }

      // Sort by confidence and show top results
      final sortedResults = medicalResults.entries.toList()
        ..sort((a, b) => b.value.compareTo(a.value));

      final buffer = StringBuffer();
      buffer.writeln("🧪 **Medical Test Strip Analysis**\n");

      // Add quality warnings if any
      if (qualityWarnings.isNotEmpty) {
        buffer.writeln("⚠️ **Image Quality Warnings:**");
        for (final warning in qualityWarnings) {
          buffer.writeln("• $warning");
        }
        buffer.writeln();
      }

      // Check overall confidence
      final maxConfidence = output[0].reduce((a, b) => a > b ? a : b);
      if (maxConfidence < 0.5) {
        buffer.writeln("⚠️ **Low Confidence Warning:** Results may be unreliable. Ensure test strip is clearly visible and well-lit.\n");
      }

      // Show parameters above 30% confidence
      final significantResults = sortedResults.where((entry) => entry.value > 0.3).toList();

      if (significantResults.isNotEmpty) {
        buffer.writeln("📊 **Detected Parameters:**");
        for (final entry in significantResults) {
          final percentage = (entry.value * 100).toStringAsFixed(1);
          final medicalValue = _getMedicalValue(entry.key, entry.value);
          final isNormal = _isNormalValue(entry.key, entry.value);
          final urgency = _getUrgencyLevel(entry.key, entry.value);

          String urgencyIcon = "🟢";
          if (urgency == "high") urgencyIcon = "🔴";
          else if (urgency == "medium") urgencyIcon = "🟡";

          buffer.writeln("$urgencyIcon **${entry.key}**: $medicalValue (${percentage}%)");

          if (!isNormal) {
            final action = _getRecommendedAction(entry.key, entry.value);
            buffer.writeln("  → $action");
          }
        }
      } else {
        // Show top result even if below threshold
        final topResult = sortedResults.first;
        final percentage = (topResult.value * 100).toStringAsFixed(1);
        final medicalValue = _getMedicalValue(topResult.key, topResult.value);
        buffer.writeln("📊 **Top Detection:**");
        buffer.writeln("• **${topResult.key}**: $medicalValue (${percentage}%)");
        buffer.writeln("\n⚠️ Very low confidence - please retake image with better lighting and focus");
      }

      // Add overall assessment
      final riskLevel = _assessOverallRisk(medicalResults);
      buffer.writeln("\n🏥 **Overall Assessment:** $riskLevel");

      buffer.writeln("\n📋 **Important:** This is for informational purposes only. Consult a healthcare provider for medical advice.");

      return buffer.toString();

    } catch (e) {
      _log("❌ Inference failed: $e");
      return "Prediction error: ${e.toString()}";
    }
  }

  List<String> _checkBasicQuality(img.Image image) {
    final warnings = <String>[];

    if (image.width < 200 || image.height < 200) {
      warnings.add("Image too small - recommend 400x400 minimum");
    }

    final aspectRatio = image.width / image.height;
    if (aspectRatio < 0.3 || aspectRatio > 3.0) {
      warnings.add("Unusual aspect ratio - may not be a test strip");
    }

    return warnings;
  }

  String _getMedicalValue(String parameter, double confidence) {
    switch (parameter) {
      case 'Glucose':
        if (confidence > 0.8) return "High (500+ mg/dL)";
        if (confidence > 0.6) return "Moderate (250 mg/dL)";
        if (confidence > 0.4) return "Trace";
        return "Negative";

      case 'Protein':
        if (confidence > 0.8) return "High (3-4+)";
        if (confidence > 0.6) return "Moderate (2+)";
        if (confidence > 0.4) return "Trace (1+)";
        return "Negative";

      case 'Blood':
        if (confidence > 0.7) return "Large";
        if (confidence > 0.5) return "Moderate";
        if (confidence > 0.3) return "Trace";
        return "Negative";

      case 'Ketones':
        if (confidence > 0.8) return "Large";
        if (confidence > 0.6) return "Moderate";
        if (confidence > 0.4) return "Trace";
        return "Negative";

      case 'pH':
        if (confidence > 0.8) return "Alkaline (8-9)";
        if (confidence > 0.6) return "Neutral (7)";
        if (confidence > 0.4) return "Slightly Acidic (6)";
        return "Acidic (5)";

      case 'Leukocytes':
        if (confidence > 0.7) return "Large";
        if (confidence > 0.5) return "Moderate";
        if (confidence > 0.3) return "Small";
        return "Negative";

      case 'Nitrite':
        return confidence > 0.5 ? "Positive" : "Negative";

      case 'Specific Gravity':
        if (confidence > 0.8) return "High (1.030)";
        if (confidence > 0.6) return "Normal (1.020)";
        if (confidence > 0.4) return "Normal (1.015)";
        return "Low (1.005)";

      case 'Bilirubin':
        if (confidence > 0.7) return "Large";
        if (confidence > 0.5) return "Moderate";
        if (confidence > 0.3) return "Small";
        return "Negative";

      case 'Urobilinogen':
        if (confidence > 0.7) return "High (8+ mg/dL)";
        if (confidence > 0.5) return "Elevated (4 mg/dL)";
        if (confidence > 0.3) return "Slightly High (2 mg/dL)";
        return "Normal";

      default:
        return "Unknown";
    }
  }

  bool _isNormalValue(String parameter, double confidence) {
    switch (parameter) {
      case 'Glucose':
      case 'Protein':
      case 'Blood':
      case 'Ketones':
      case 'Leukocytes':
      case 'Nitrite':
      case 'Bilirubin':
        return confidence <= 0.3;

      case 'pH':
        return confidence > 0.3 && confidence < 0.8; // Normal range

      case 'Specific Gravity':
        return confidence > 0.3 && confidence < 0.8; // Normal range

      case 'Urobilinogen':
        return confidence <= 0.3;

      default:
        return true;
    }
  }

  String _getUrgencyLevel(String parameter, double confidence) {
    if (!_isNormalValue(parameter, confidence)) {
      switch (parameter) {
        case 'Protein':
        case 'Ketones':
        case 'Bilirubin':
          return "high";
        case 'Glucose':
        case 'Blood':
        case 'Leukocytes':
        case 'Nitrite':
          return "medium";
        default:
          return "low";
      }
    }
    return "low";
  }

  String _getRecommendedAction(String parameter, double confidence) {
    switch (parameter) {
      case 'Protein':
        if (confidence > 0.7) return "Urgent kidney function evaluation needed";
        return "Monitor kidney function, repeat test";

      case 'Glucose':
        if (confidence > 0.7) return "Evaluate for diabetes and kidney function";
        return "Monitor blood glucose, repeat test";

      case 'Ketones':
        if (confidence > 0.7) return "Seek immediate medical attention";
        return "Monitor closely, check blood glucose";

      case 'Blood':
        return "Evaluate for kidney stones, infection, or injury";

      case 'Bilirubin':
        return "Evaluate liver function and kidney involvement";

      case 'Leukocytes':
      case 'Nitrite':
        return "Evaluate for urinary tract infection";

      default:
        return "Monitor and consult healthcare provider if persistent";
    }
  }

  String _assessOverallRisk(Map<String, double> results) {
    int highRiskCount = 0;
    int moderateRiskCount = 0;

    results.forEach((parameter, confidence) {
      if (!_isNormalValue(parameter, confidence)) {
        final urgency = _getUrgencyLevel(parameter, confidence);
        if (urgency == "high") highRiskCount++;
        else if (urgency == "medium") moderateRiskCount++;
      }
    });

    if (highRiskCount >= 2) {
      return "🚨 HIGH RISK - Multiple serious abnormalities detected";
    } else if (highRiskCount >= 1 || moderateRiskCount >= 2) {
      return "⚠️ MODERATE RISK - Abnormalities require medical follow-up";
    } else if (moderateRiskCount >= 1) {
      return "📋 LOW RISK - Minor abnormalities, monitor closely";
    } else {
      return "✅ NORMAL - No significant abnormalities detected";
    }
  }

  img.Image processImage(img.Image image) {
    final double inputAspect = image.width / image.height;
    final double targetAspect = 1.0;
    int resizeWidth, resizeHeight;

    if (inputAspect > targetAspect) {
      resizeWidth = inputSize;
      resizeHeight = (inputSize / inputAspect).round();
    } else {
      resizeHeight = inputSize;
      resizeWidth = (inputSize * inputAspect).round();
    }

    final img.Image resized = img.copyResize(
      image,
      width: resizeWidth,
      height: resizeHeight,
      interpolation: img.Interpolation.linear,
    );

    final img.Image result = img.Image.rgb(inputSize, inputSize);
    final int offsetX = (inputSize - resized.width) ~/ 2;
    final int offsetY = (inputSize - resized.height) ~/ 2;

    // ✅ Fill background with zeros (black) for padding
    img.fill(result, img.getColor(0, 0, 0));

    for (int y = 0; y < resized.height; y++) {
      for (int x = 0; x < resized.width; x++) {
        result.setPixel(offsetX + x, offsetY + y, resized.getPixel(x, y));
      }
    }

    return result;
  }

  Float32List imageToBuffer(img.Image image) {
    final buffer = Float32List(inputSize * inputSize * 3);
    int bufferIndex = 0;

    for (int y = 0; y < inputSize; y++) {
      for (int x = 0; x < inputSize; x++) {
        final pixel = image.getPixel(x, y);
        final r = ((pixel >> 16) & 0xFF) / 255.0;
        final g = ((pixel >> 8) & 0xFF) / 255.0;
        final b = (pixel & 0xFF) / 255.0;

        buffer[bufferIndex++] = r;
        buffer[bufferIndex++] = g;
        buffer[bufferIndex++] = b;
      }
    }

    return buffer;
  }

  void dispose() {
    if (_isInitialized) {
      interpreter.close();
      _isInitialized = false;
    }
  }
}