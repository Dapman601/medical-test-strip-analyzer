import 'dart:convert';
import 'package:flutter/services.dart';

class DiseaseLabels {
  final Map<int, String> _labels = {}; // Make this 'final'

  /// Load disease labels from a JSON file.
  Future<void> loadLabels(String path) async {
    try {
      // Load the JSON file from assets
      final data = await rootBundle.loadString(path);

      // Parse the JSON data
      final jsonResult = json.decode(data);

      // Ensure the categories field exists in the JSON
      final categories = jsonResult['categories'] as List?;
      if (categories == null) {
        throw FormatException("The JSON does not contain a valid 'categories' field.");
      }

      // Map category ids to their names
      for (var category in categories) {
        if (category['id'] != null && category['name'] != null) {
          _labels[category['id']] = category['name'];
        }
      }
    } catch (e) {
      // Instead of print, throw an exception or use logging
      throw Exception("Error loading labels: $e");
    }
  }

  /// Get the label corresponding to the given index.
  String getLabel(int index) {
    return _labels[index] ?? "Unknown";
  }

  /// Get the total number of labels.
  int getLabelsCount() {
    return _labels.length;
  }

  /// Get all labels as a list.
  List<String> getAllLabels() {
    return _labels.values.toList();
  }
}