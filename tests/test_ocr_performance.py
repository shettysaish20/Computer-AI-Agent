import sys
import os
import time
import glob
import tempfile
import shutil
import argparse
from PIL import Image

# Add parent directory to path so we can import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ocr_detector import OCRDetector, OCRDetConfig
from utils.beautiful_visualizer import BeautifulVisualizer

def get_test_images():
    """Automatically discover all images in the images folder"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
    images_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images')
    
    all_images = []
    for ext in image_extensions:
        pattern = os.path.join(images_dir, ext)
        all_images.extend(glob.glob(pattern))
        # Also check uppercase extensions
        pattern = os.path.join(images_dir, ext.upper())
        all_images.extend(glob.glob(pattern))
    
    return sorted(list(set(all_images)))  # Remove duplicates and sort

def create_batch_test_images(image_paths, target_size=(1920, 1280)):
    """
    Create temporary resized copies of images for batch testing
    All images will be exactly the same size for batch compatibility
    """
    temp_dir = tempfile.mkdtemp(prefix='ocr_batch_test_')
    temp_images = []
    
    print(f"📏 Creating batch test images ({target_size[0]}x{target_size[1]})...")
    
    for i, img_path in enumerate(image_paths):
        try:
            # Load and resize to exact target size
            img = Image.open(img_path).convert("RGB")
            img_resized = img.resize(target_size)
            
            # Save to temp directory
            base_name = f"batch_{i:03d}_{os.path.basename(img_path)}"
            temp_path = os.path.join(temp_dir, base_name)
            img_resized.save(temp_path)
            temp_images.append(temp_path)
            
        except Exception as e:
            print(f"⚠️  Error processing {img_path}: {e}")
    
    print(f"✅ Created {len(temp_images)} batch-ready images in {temp_dir}")
    return temp_images, temp_dir

def create_visualization_if_enabled(image_path, detections, show_flag, test_name="test"):
    """Create and save visualization if --show flag is enabled"""
    if not show_flag:
        return
    
    try:
        # Initialize visualizer
        visualizer = BeautifulVisualizer(output_dir="outputs")
        
        # Prepare results in the format expected by the visualizer
        results = {
            'ocr_detections': detections
        }
        
        # Create filename base
        filename_base = f"ocr_{test_name}_{os.path.splitext(os.path.basename(image_path))[0]}"
        
        # Create visualization
        viz_paths = visualizer.create_all_visualizations(
            image_path=image_path,
            results=results,
            filename_base=filename_base
        )
        
        print(f"🎨 OCR Visualizations saved:")
        for viz_type, path in viz_paths.items():
            print(f"   📷 {viz_type.upper()}: {os.path.basename(path)}")
            
        return viz_paths
        
    except Exception as e:
        print(f"⚠️  OCR Visualization error: {e}")
        return None

def test_single_image(original_images, show_flag=False):
    """Test single image OCR processing"""
    print("\n1️⃣ Single Image OCR Processing Test")
    config = OCRDetConfig(enable_timing=True)
    detector = OCRDetector(config)
    
    if original_images:
        start = time.time()
        detections = detector.detect(original_images[0])
        end = time.time()
        print(f"   {os.path.basename(original_images[0])}: {len(detections)} OCR detections in {end-start:.3f}s")
        
        # Create visualization if requested
        if show_flag:
            create_visualization_if_enabled(
                original_images[0], detections, show_flag, "single_image"
            )
        
        return detector, detections
    return None, []

def test_batch_processing(original_images, show_flag=False):
    """Test batch OCR processing"""
    print(f"\n2️⃣ Batch OCR Processing Test ({len(original_images)} images)")
    
    config = OCRDetConfig(enable_timing=True)
    detector = OCRDetector(config)
    
    # OCR doesn't have built-in batch processing, so we'll process individually
    start = time.time()
    batch_results = []
    
    for i, img_path in enumerate(original_images):
        print(f"   Processing image {i+1}/{len(original_images)}: {os.path.basename(img_path)}")
        detections = detector.detect(img_path)
        batch_results.append(detections)
    
    end = time.time()
    total_detections = sum(len(dets) for dets in batch_results)
    print(f"   Batch: {total_detections} total OCR detections in {end-start:.3f}s")
    print(f"   Per image: {(end-start)/len(original_images):.3f}s average")
    
    print(f"\n📊 Per-Image OCR Results:")
    for i, (orig_path, dets) in enumerate(zip(original_images, batch_results)):
        orig_name = os.path.basename(orig_path)
        print(f"   {i+1:2d}. {orig_name:<20} {len(dets):3d} OCR detections")
    
    # Create visualizations for first few images if requested
    if show_flag:
        print(f"\n🎨 Creating batch OCR visualizations...")
        for i, (orig_path, dets) in enumerate(zip(original_images[:3], batch_results[:3])):
            create_visualization_if_enabled(
                orig_path, dets, show_flag, f"batch_{i+1:02d}"
            )
    
    return detector, batch_results

def test_model_cache(detector, original_images, show_flag=False):
    """Test OCR model cache functionality"""
    print("\n3️⃣ OCR Model Cache Test")
    start = time.time()
    # OCR detector doesn't have reset cache method, so we create a new detector
    new_config = OCRDetConfig(enable_timing=True)
    new_detector = OCRDetector(new_config)
    detections = new_detector.detect(original_images[0])
    end = time.time()
    print(f"   New detector + detection: {end-start:.3f}s")
    
    # Create visualization if requested
    if show_flag:
        create_visualization_if_enabled(
            original_images[0], detections, show_flag, "cache_test"
        )

def test_threshold_comparison(original_images, show_flag=False):
    """Test OCR detection threshold comparison"""
    print("\n4️⃣ OCR Threshold Comparison Test")
    
    # Low threshold (more sensitive)
    config_low = OCRDetConfig(det_threshold=0.1, enable_timing=False)
    detector_low = OCRDetector(config_low)
    
    # Standard threshold  
    config_std = OCRDetConfig(det_threshold=0.3, enable_timing=False)
    detector_std = OCRDetector(config_std)
    
    # High threshold (less sensitive)
    config_high = OCRDetConfig(det_threshold=0.6, enable_timing=False)
    detector_high = OCRDetector(config_high)
    
    dets_low = detector_low.detect(original_images[0])
    dets_std = detector_std.detect(original_images[0])
    dets_high = detector_high.detect(original_images[0])
    
    print(f"   Low threshold (0.1): {len(dets_low)} detections")
    print(f"   Standard threshold (0.3): {len(dets_std)} detections")
    print(f"   High threshold (0.6): {len(dets_high)} detections")
    
    # Create visualizations for different thresholds if requested
    if show_flag:
        print(f"\n🎨 Creating threshold comparison visualizations...")
        create_visualization_if_enabled(
            original_images[0], dets_low, show_flag, "low_threshold"
        )
        create_visualization_if_enabled(
            original_images[0], dets_std, show_flag, "std_threshold"
        )
        create_visualization_if_enabled(
            original_images[0], dets_high, show_flag, "high_threshold"
        )

def test_dilation_comparison(original_images, show_flag=False):
    """Test OCR dilation on/off comparison"""
    print("\n5️⃣ OCR Dilation Comparison Test")
    
    config_no_dilation = OCRDetConfig(use_dilation=False, enable_timing=False)
    detector_no_dilation = OCRDetector(config_no_dilation)
    
    config_with_dilation = OCRDetConfig(use_dilation=True, enable_timing=False)
    detector_with_dilation = OCRDetector(config_with_dilation)
    
    dets_no_dilation = detector_no_dilation.detect(original_images[0])
    dets_with_dilation = detector_with_dilation.detect(original_images[0])
    
    print(f"   Without dilation: {len(dets_no_dilation)} detections")
    print(f"   With dilation: {len(dets_with_dilation)} detections")
    print(f"   Dilation effect: +{len(dets_with_dilation) - len(dets_no_dilation)} detections")
    
    # Create visualizations for both versions if requested
    if show_flag:
        print(f"\n🎨 Creating dilation comparison visualizations...")
        create_visualization_if_enabled(
            original_images[0], dets_no_dilation, show_flag, "no_dilation"
        )
        create_visualization_if_enabled(
            original_images[0], dets_with_dilation, show_flag, "with_dilation"
        )

def test_resolution_comparison(original_images, show_flag=False):
    """Test OCR max resolution comparison"""
    if len(original_images) < 3:
        print(f"\n6️⃣ OCR Resolution Comparison: Skipped (need 3+ images, have {len(original_images)})")
        return
        
    print(f"\n6️⃣ OCR Resolution Comparison (first image)")
    test_image = original_images[0]
    
    # Low resolution
    config_low_res = OCRDetConfig(max_side_len=480, enable_timing=False)
    detector_low_res = OCRDetector(config_low_res)
    
    # Standard resolution
    config_std_res = OCRDetConfig(max_side_len=960, enable_timing=False)
    detector_std_res = OCRDetector(config_std_res)
    
    # High resolution
    config_high_res = OCRDetConfig(max_side_len=1440, enable_timing=False)
    detector_high_res = OCRDetector(config_high_res)
    
    # Time each resolution
    start = time.time()
    dets_low_res = detector_low_res.detect(test_image)
    time_low = time.time() - start
    
    start = time.time()
    dets_std_res = detector_std_res.detect(test_image)
    time_std = time.time() - start
    
    start = time.time()
    dets_high_res = detector_high_res.detect(test_image)
    time_high = time.time() - start
    
    print(f"   Low res (480): {len(dets_low_res)} detections in {time_low:.3f}s")
    print(f"   Std res (960): {len(dets_std_res)} detections in {time_std:.3f}s")
    print(f"   High res (1440): {len(dets_high_res)} detections in {time_high:.3f}s")
    
    # Create comparison visualizations if requested
    if show_flag:
        print(f"\n🎨 Creating resolution comparison visualizations...")
        create_visualization_if_enabled(
            test_image, dets_low_res, show_flag, "low_res_480"
        )
        create_visualization_if_enabled(
            test_image, dets_std_res, show_flag, "std_res_960"
        )
        create_visualization_if_enabled(
            test_image, dets_high_res, show_flag, "high_res_1440"
        )

def main():
    """Main function with CLI arguments"""
    parser = argparse.ArgumentParser(
        description="OCR Performance Testing Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test Modes:
  single    - Single image OCR processing test (default)
  batch     - Batch OCR processing test  
  cache     - Model cache test
  threshold - Detection threshold comparison
  dilation  - Dilation on/off comparison
  resolution - Resolution comparison
  all       - Run all tests

Examples:
  python test_ocr_performance.py                      # Single image test only
  python test_ocr_performance.py --mode batch         # Batch test only
  python test_ocr_performance.py --mode all           # All tests
  python test_ocr_performance.py --show               # Single test with visualization
  python test_ocr_performance.py --mode all --show    # All tests with visualizations
  python test_ocr_performance.py -m threshold dilation --show # Specific tests with viz
        """
    )
    
    parser.add_argument(
        "--mode", "-m",
        nargs="+",
        choices=["single", "batch", "cache", "threshold", "dilation", "resolution", "all"],
        default=["single"],
        help="Test mode(s) to run (default: single)"
    )
    
    parser.add_argument(
        "--show", "-s",
        action="store_true",
        help="Create and save beautiful visualizations of OCR detection results (saved to outputs/ folder)"
    )
    
    args = parser.parse_args()
    
    # Discover test images
    original_images = get_test_images()
    
    if not original_images:
        print("❌ No images found in 'images' folder!")
        print("   Place some test images (.jpg, .png, etc.) in the images/ directory")
        return
    
    print(f"🚀 Testing OCR Detection Optimizations with {len(original_images)} images")
    print(f"📁 Found images: {[os.path.basename(img) for img in original_images[:5]]}")
    if len(original_images) > 5:
        print(f"   ... and {len(original_images) - 5} more")
    print(f"🎯 Running modes: {', '.join(args.mode)}")
    if args.show:
        print(f"🎨 Visualization: ENABLED (outputs saved to outputs/ folder)")
    print("=" * 50)
    
    detector = None
    
    # Expand "all" mode
    if "all" in args.mode:
        modes = ["single", "batch", "cache", "threshold", "dilation", "resolution"]
    else:
        modes = args.mode
    
    # Run selected tests
    for mode in modes:
        if mode == "single":
            detector, _ = test_single_image(original_images, args.show)
        elif mode == "batch":
            test_batch_processing(original_images, args.show)
        elif mode == "cache":
            if detector is None:
                detector = OCRDetector(OCRDetConfig(enable_timing=True))
            test_model_cache(detector, original_images, args.show)
        elif mode == "threshold":
            test_threshold_comparison(original_images, args.show)
        elif mode == "dilation":
            test_dilation_comparison(original_images, args.show)
        elif mode == "resolution":
            test_resolution_comparison(original_images, args.show)
    
    if args.show:
        print(f"\n✅ All OCR visualizations saved to: outputs/ folder")
        print(f"🔍 Check the outputs/ directory for beautiful OCR detection visualizations!")

if __name__ == "__main__":
    main()


# Usage Examples:

# # Default: Single image OCR test only (no visualization)
# uv run tests/test_ocr_performance.py

# # Single OCR test with beautiful visualization
# uv run tests/test_ocr_performance.py --show

# # Specific modes with visualization
# uv run tests/test_ocr_performance.py --mode batch --show
# uv run tests/test_ocr_performance.py --mode threshold --show
# uv run tests/test_ocr_performance.py --mode dilation --show

# # Multiple modes with visualization
# uv run tests/test_ocr_performance.py -m single threshold dilation --show
# uv run tests/test_ocr_performance.py -m batch resolution --show

# # All OCR tests with beautiful visualizations
# uv run tests/test_ocr_performance.py --mode all --show

# # Help
# uv run tests/test_ocr_performance.py --help
