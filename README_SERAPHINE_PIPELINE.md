# 🚀 Seraphine AI Pipeline V1.2

## Detection → Merging → Seraphine Grouping → Gemini Analysis → Export

A comprehensive AI-powered pipeline for intelligent UI element detection, grouping, and analysis. Combines YOLO object detection, OCR text recognition, smart bbox merging, geometric grouping (Seraphine), and LLM-powered icon analysis (Gemini).

---

## 🌟 Features

### 🎯 **Dual Operation Modes**
- **Debug Mode**: Full verbose output with visualizations, JSONs, and detailed logs
- **Deploy MCP Mode**: Silent operation with minimal output, perfect for API integration

### 🔧 **Advanced Detection Pipeline**
- **Parallel Processing**: YOLO + OCR run simultaneously for maximum speed
- **Intelligent Merging**: 3-stage bbox merging with overlap resolution
- **Smart Filtering**: Automatically removes YOLO boxes containing >2 OCR elements

### 🧠 **Seraphine Intelligent Grouping**
- **Geometric Analysis**: Groups UI elements by spatial relationships
- **Layout Understanding**: Identifies horizontal, vertical, and long-box patterns
- **Overlap Resolution**: Handles complex UI layouts with overlapping elements

### 🤖 **Gemini LLM Integration**
- **Icon Recognition**: AI-powered identification of UI elements
- **Batch Processing**: Optimized concurrent API calls
- **Rich Descriptions**: Detailed analysis of each detected element

### 📊 **Complete Traceability**
- **ID Tracking**: Y001 → M001 → H1_1 → Gemini analysis
- **Perfect Mapping**: Every element tracked through entire pipeline
- **Comprehensive Output**: JSON + visualizations + performance metrics

---

## 🛠️ Installation

### Prerequisites
```bash
# Python 3.8+
python --version

# UV package manager (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup
```bash
# Clone repository
git clone <repository-url>
cd enhanced-ai-pipeline

# Install dependencies
uv sync

# Download models (place in models/ directory)
# - models/model_dynamic.onnx (YOLO)
# - models/ch_PP-OCRv3_det_infer.onnx (OCR)

# Setup Gemini API key
export GOOGLE_API_KEY="your-gemini-api-key"
```

### Directory Structure
```
enhanced-ai-pipeline/
├── images/                    # Input images
│   └── word.png              # Example image
├── models/                    # AI models
│   ├── model_dynamic.onnx     # YOLO detection model
│   └── ch_PP-OCRv3_det_infer.onnx # OCR detection model
├── outputs/                   # Generated outputs (auto-managed)
├── utils/                     # Core pipeline modules
│   ├── config.json           # Configuration file
│   ├── prompt.txt            # Gemini analysis prompt
│   ├── yolo_detector.py      # YOLO detection
│   ├── ocr_detector.py       # OCR detection
│   ├── bbox_merger.py        # Intelligent merging
│   ├── seraphine_processor.py # Geometric grouping
│   ├── gemini_integration.py # LLM analysis
│   ├── beautiful_visualizer.py # Visualization creation
│   └── parallel_processor.py # Concurrent processing
├── main.py                   # Main pipeline entry point
└── README.md                 # This file
```

Detection + Merge:     2.5-3.0s
Seraphine Grouping:    0.03-0.08s  
Image Generation:      0.2-0.3s
Gemini Analysis:       10-15s
Visualizations:        0.3-0.5s
Total Pipeline:        13-19s

---

## 🐛 Troubleshooting

### Common Issues

**Silent Failure in Deploy Mode:**
```bash
# Check if image exists
ls images/word.png

# Verify models are present
ls models/

# Check Gemini API key
echo $GOOGLE_API_KEY
```

**ONNX Model Errors:**
```bash
# Ensure correct model versions
# YOLO: model_dynamic.onnx (YOLOv8 format)
# OCR: ch_PP-OCRv3_det_infer.onnx (PaddleOCR format)
```

**Memory Issues:**
```bash
# Reduce OCR max_side_len in config
"ocr_max_side_len": 960

# Reduce Gemini concurrent requests
"gemini_max_concurrent": 2
```

### Debug Mode Testing
```bash
# Set debug mode
# Edit utils/config.json: "mode": "debug"

# Run pipeline
uv run main.py

# Check outputs
ls outputs/
```

### Deploy Mode Testing
```bash
# Set deploy mode  
# Edit utils/config.json: "mode": "deploy_mcp"

# Run pipeline
uv run main.py

# Should only output: "Pipeline completed in X.Xs, found Y icons."
```

---

## 🤝 Contributing

1. **Fork** the repository
2. **Create** your feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **YOLO**: Object detection framework
- **PaddleOCR**: Text detection and recognition
- **Google Gemini**: Advanced AI analysis
- **OpenCV**: Computer vision operations
- **PIL/Pillow**: Image processing

---

## 📞 Support

For questions, issues, or contributions:
- **GitHub Issues**: [Create an issue](https://github.com/your-repo/issues)
- **Documentation**: This README
- **Examples**: See `examples/` directory

---

**🚀 Ready to detect, group, and analyze! Happy coding!**

---

## ⚙️ Configuration

### `utils/config.json`
```json
{
    "mode": "debug",                    // "debug" | "deploy_mcp"
    
    // Core Settings
    "save_json": true,
    "save_visualizations": true,
    "generate_grouped_images": true,
    "output_dir": "outputs",
    
    // YOLO Configuration
    "yolo_model_path": "models/model_dynamic.onnx",
    "yolo_conf_threshold": 0.1,
    "yolo_iou_threshold": 0.1,
    "yolo_enable_timing": true,
    "yolo_enable_debug": true,
    
    // OCR Configuration
    "ocr_model_path": "models/ch_PP-OCRv3_det_infer.onnx",
    "ocr_det_threshold": 0.9,
    "ocr_max_side_len": 1280,
    "ocr_enable_timing": true,
    "ocr_enable_debug": true,
    
    // Merging Configuration
    "merger_iou_threshold": 0.05,
    
    // Seraphine Configuration
    "seraphine_timing": false,
    "seraphine_enable_debug": false,
    
    // Gemini Configuration
    "gemini_enabled": true,
    "gemini_prompt_path": "utils/prompt.txt",
    "gemini_return_images_b64": true,
    "gemini_max_concurrent": 4,
    
    // Visualization Settings
    "save_gemini_visualization": true,
    "save_yolo_viz": true,
    "save_ocr_viz": true,
    "save_merged_viz": true,
    "save_complete_viz": true,
    "save_seraphine_viz": true
}
```

---

## 🚀 Usage

### 🐛 Debug Mode (Development)

**Configuration:**
```json
{
    "mode": "debug"
}
```

**Run:**
```bash
uv run main.py
```

**Output:**
```bash
🚀 ENHANCED AI PIPELINE V1.2: Detection + Merging + Seraphine + Gemini + Export
==========================================================================================
✅ Configuration loaded from config.json
🎯 YOLO Config: conf=0.1, iou=0.1
📝 OCR Config: threshold=0.9, max_len=1280
📸 Image loaded: 1920x1080 pixels

🔄 Step 1: Parallel YOLO + OCR Detection + Intelligent Merging (FIXED)
============================================================
⚡ Parallel detection completed in 2.540s
  YOLO found: 124 detections
  OCR found: 78 detections

🧠 Step 2: Seraphine Intelligent Grouping & Layout Analysis
============================================================
📊 Seraphine Grouping Results:
  🧠 Input: 159 merged detections (M001-M159)
  📦 Groups created: 75
  📐 Horizontal groups: 65
  📏 Vertical groups: 10

🤖 Step 4: Gemini LLM Analysis
======================================================================
🤖 GEMINI ANALYSIS SUMMARY:
  📊 Images analyzed: 3/3
  🎯 Total icons found: 154
  ⏱️  Analysis time: 11.85s

📊 ENHANCED PIPELINE V1.2 SUMMARY:
=================================================================
  ⏱️  Total pipeline time: 14.634s
  🤖 GEMINI analysis: ✅ 3/3 images analyzed
     🎯 Total icons found: 154
  💾 Enhanced JSON: word_enhanced_v1_01-06.json
  🎨 Visualizations: 6 types created
```

**Generated Files:**
```bash
outputs/
├── v1_word_yolo_result_11-07.jpg           # YOLO detections
├── v1_word_ocr_result_11-07.jpg            # OCR detections
├── v1_word_merged_result_11-07.jpg         # Merged detections
├── v1_word_complete_result_11-07.jpg       # All detections
├── v1_word_seraphine_groups_11-07.jpg      # Intelligent groups
├── v1_word_gemini_analysis_11-07.jpg       # Gemini labeled
├── word_enhanced_v1_01-06.json             # Complete pipeline data
└── word_gemini_analysis_11-07.json         # Gemini analysis
```

### 🚀 Deploy MCP Mode (Production)

**Configuration:**
```json
{
    "mode": "deploy_mcp"
}
```

**Run:**
```bash
uv run main.py
```

**Output:**
```bash
Pipeline completed in 12.345s, found 154 icons.
```

**Return Data:**
```python
{
    'total_time': 12.345,
    'total_icons_found': 154,
    'seraphine_gemini_groups': {
        'H1_1': {
            'bbox': [10, 20, 100, 30],
            'g_icon_name': 'Weather Widget',
            'g_brief': 'Shows current temperature and weather conditions',
            'm_id': 'M001',
            'y_id': 'Y001',
            'o_id': 'NA'
        },
        'H1_2': {...},
        'V1_1': {...}
        // ... all grouped elements with Gemini analysis
    }
}
```

---

## 🔧 API Reference

### Main Function
```python
import asyncio
from main import main

# Run pipeline
results = asyncio.run(main())

if results:
    # Access results
    if 'seraphine_gemini_groups' in results:
        groups = results['seraphine_gemini_groups']
        total_time = results['total_time']
        icon_count = results['total_icons_found']
    
    # Debug mode additional data
    if 'detection_results' in results:
        yolo_detections = results['detection_results']['yolo_detections']
        ocr_detections = results['detection_results']['ocr_detections']
        merged_detections = results['detection_results']['merged_detections']
```

### Individual Components
```python
from utils.yolo_detector import YOLODetector, YOLOConfig
from utils.ocr_detector import OCRDetector, OCRDetConfig
from utils.bbox_merger import BBoxMerger
from utils.seraphine_processor import FinalSeraphineProcessor
from utils.gemini_integration import run_gemini_analysis

# Initialize detectors
yolo_config = YOLOConfig(
    model_path="models/model_dynamic.onnx",
    conf_threshold=0.1,
    iou_threshold=0.1
)
yolo_detector = YOLODetector(yolo_config)

ocr_config = OCRDetConfig(
    model_path="models/ch_PP-OCRv3_det_infer.onnx",
    det_threshold=0.9,
    max_side_len=1280
)
ocr_detector = OCRDetector(ocr_config)

# Run detections
yolo_results = yolo_detector.detect("image.jpg")
ocr_results = ocr_detector.detect("image.jpg")

# Merge results
merger = BBoxMerger()
merged_results, stats = merger.merge_detections(yolo_results, ocr_results)

# Group with Seraphine
processor = FinalSeraphineProcessor()
seraphine_analysis = processor.analyze(merged_results)

# Analyze with Gemini
gemini_results = await run_gemini_analysis(seraphine_analysis, grouped_images, "image.jpg", config)
```

---

## 📊 Output Format

### Debug Mode Return
```python
{
    'detection_results': {
        'yolo_detections': [...],      # Raw YOLO detections
        'ocr_detections': [...],       # Raw OCR detections  
        'merged_detections': [...],    # Intelligently merged
        'timing': {...},               # Performance metrics
        'merge_stats': {...}           # Merge statistics
    },
    'seraphine_analysis': {
        'analysis': {...},             # Group statistics
        'seraphine_gemini_groups': {...} # Final grouped data
    },
    'gemini_results': {
        'total_icons_found': 154,     # Total icons
        'analysis_duration_seconds': 11.85,
        'successful_analyses': 3,
        'icons': [...]                 # Detailed icon data
    },
    'grouped_image_paths': [...],      # Generated group images
    'visualization_paths': {...},     # All visualization files
    'json_path': "outputs/enhanced.json",
    'config': {...},                   # Used configuration
    'total_time': 14.634
}
```

### Deploy MCP Mode Return
```python
{
    'total_time': 12.345,
    'total_icons_found': 154,
    'seraphine_gemini_groups': {
        'H1_1': {
            'bbox': [x, y, w, h],
            'g_icon_name': 'Icon Name',
            'g_brief': 'Description',
            'm_id': 'M001',
            'y_id': 'Y001',
            'o_id': 'NA'
        }
        // ... more groups
    }
}
```

---

## 🎯 Performance Optimizations

### ⚡ Speed Improvements
- **Parallel Processing**: YOLO + OCR run simultaneously
- **ONNX Models**: Optimized inference engines
- **Concurrent Gemini**: Multiple API calls in parallel
- **Memory Optimization**: PIL images passed directly to detectors
- **Smart Caching**: Model loading optimized

### 📈 Typical Performance
