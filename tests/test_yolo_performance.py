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

from utils.yolo_detector import YOLODetector, YOLOConfig
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
    temp_dir = tempfile.mkdtemp(prefix='yolo_batch_test_')
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
            'yolo_detections': detections
        }
        
        # Create filename base
        filename_base = f"{test_name}_{os.path.splitext(os.path.basename(image_path))[0]}"
        
        # Create visualization
        viz_paths = visualizer.create_all_visualizations(
            image_path=image_path,
            results=results,
            filename_base=filename_base
        )
        
        print(f"🎨 Visualizations saved:")
        for viz_type, path in viz_paths.items():
            print(f"   📷 {viz_type.upper()}: {os.path.basename(path)}")
            
        return viz_paths
        
    except Exception as e:
        print(f"⚠️  Visualization error: {e}")
        return None

def test_single_image(original_images, show_flag=False):
    """Test single image processing"""
    print("\n1️⃣ Single Image Processing Test")
    config = YOLOConfig(enable_timing=True)
    detector = YOLODetector(config)
    
    if original_images:
        start = time.time()
        detections = detector.detect(original_images[2])
        end = time.time()
        print(f"   {os.path.basename(original_images[2])}: {len(detections)} detections in {end-start:.3f}s")
        
        # Create visualization if requested
        if show_flag:
            create_visualization_if_enabled(
                original_images[2], detections, show_flag, "single_image"
            )
        
        return detector, detections
    return None, []

def test_batch_processing(original_images, show_flag=False):
    """Test batch processing with resized images"""
    print(f"\n2️⃣ Batch Processing Test ({len(original_images)} images @ 1920x1280)")
    
    batch_images, temp_dir = create_batch_test_images(original_images, target_size=(1920, 1280))
    
    try:
        batch_config = YOLOConfig(enable_timing=True, max_resolution=(1920, 1280))
        batch_detector = YOLODetector(batch_config)
        
        start = time.time()
        batch_results = batch_detector.detect_batch(batch_images)
        end = time.time()
        total_detections = sum(len(dets) for dets in batch_results)
        print(f"   Batch: {total_detections} total detections in {end-start:.3f}s")
        print(f"   Per image: {(end-start)/len(batch_images):.3f}s average")
        
        print(f"\n📊 Per-Image Results:")
        for i, (orig_path, dets) in enumerate(zip(original_images, batch_results)):
            orig_name = os.path.basename(orig_path)
            print(f"   {i+1:2d}. {orig_name:<20} {len(dets):3d} detections")
        
        # Create visualizations for first few images if requested
        if show_flag:
            print(f"\n🎨 Creating batch visualizations...")
            for i, (orig_path, dets) in enumerate(zip(original_images[:3], batch_results[:3])):
                create_visualization_if_enabled(
                    orig_path, dets, show_flag, f"batch_{i+1:02d}"
                )
        
        return batch_detector, batch_results
    
    finally:
        shutil.rmtree(temp_dir)
        print(f"🧹 Cleaned up temporary files")

def test_model_cache(detector, original_images, show_flag=False):
    """Test model cache reset functionality"""
    print("\n3️⃣ Model Cache Test")
    start = time.time()
    detector.reset_model_cache()
    detections = detector.detect(original_images[2])
    end = time.time()
    print(f"   Cache reset + reload: {end-start:.3f}s")
    
    # Create visualization if requested
    if show_flag:
        create_visualization_if_enabled(
            original_images[2], detections, show_flag, "cache_test"
        )

def test_content_filtering(original_images, show_flag=False):
    """Test content filtering comparison"""
    print("\n4️⃣ Content Filtering Test")
    config_no_filter = YOLOConfig(enable_content_filtering=False, enable_timing=False)
    detector_no_filter = YOLODetector(config_no_filter)
    
    config_with_filter = YOLOConfig(enable_content_filtering=True, enable_timing=False)
    detector_with_filter = YOLODetector(config_with_filter)
    
    dets_no_filter = detector_no_filter.detect(original_images[2])
    dets_with_filter = detector_with_filter.detect(original_images[2])
    
    filtered_out = len(dets_no_filter) - len(dets_with_filter)
    print(f"   Without filtering: {len(dets_no_filter)} detections")
    print(f"   With filtering: {len(dets_with_filter)} detections")
    print(f"   Filtered out: {filtered_out} sparse boxes")
    
    # Create visualizations for both versions if requested
    if show_flag:
        print(f"\n🎨 Creating filtering comparison visualizations...")
        create_visualization_if_enabled(
            original_images[2], dets_no_filter, show_flag, "no_filter"
        )
        create_visualization_if_enabled(
            original_images[2], dets_with_filter, show_flag, "with_filter"
        )

def test_batch_vs_individual(original_images, show_flag=False):
    """Test batch vs individual processing comparison"""
    if len(original_images) < 3:
        print(f"\n5️⃣ Batch vs Individual Comparison: Skipped (need 3+ images, have {len(original_images)})")
        return
        
    print(f"\n5️⃣ Batch vs Individual Comparison (first 3 images)")
    test_subset = original_images[:3]
    
    batch_subset, temp_dir_subset = create_batch_test_images(test_subset, target_size=(1920, 1280))
    
    try:
        # Individual processing
        detector = YOLODetector(YOLOConfig(enable_timing=False))
        detector.reset_model_cache()
        start = time.time()
        individual_results = []
        for img in test_subset:
            individual_results.append(detector.detect(img))
        individual_time = time.time() - start
        
        # Batch processing
        batch_detector = YOLODetector(YOLOConfig(enable_timing=False, max_resolution=(1920, 1280)))
        batch_detector.reset_model_cache()
        start = time.time()
        batch_subset_results = batch_detector.detect_batch(batch_subset)
        batch_subset_time = time.time() - start
        
        print(f"   Individual (3 images): {individual_time:.3f}s")
        print(f"   Batch (3 images):      {batch_subset_time:.3f}s")
        if batch_subset_time > 0:
            print(f"   Batch speedup:         {individual_time/batch_subset_time:.1f}x")
        
        # Create comparison visualizations if requested
        if show_flag:
            print(f"\n🎨 Creating comparison visualizations...")
            for i, (orig_path, ind_dets, batch_dets) in enumerate(zip(test_subset, individual_results, batch_subset_results)):
                create_visualization_if_enabled(
                    orig_path, ind_dets, show_flag, f"individual_{i+1:02d}"
                )
                create_visualization_if_enabled(
                    orig_path, batch_dets, show_flag, f"batch_comp_{i+1:02d}"
                )
    
    finally:
        shutil.rmtree(temp_dir_subset)

def main():
    """Main function with CLI arguments"""
    parser = argparse.ArgumentParser(
        description="YOLO Performance Testing Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test Modes:
  single    - Single image processing test (default)
  batch     - Batch processing test  
  cache     - Model cache test
  filter    - Content filtering comparison
  compare   - Batch vs individual comparison
  all       - Run all tests

Examples:
  python test_yolo_performance.py                    # Single image test only
  python test_yolo_performance.py --mode batch       # Batch test only
  python test_yolo_performance.py --mode all         # All tests
  python test_yolo_performance.py --show             # Single test with visualization
  python test_yolo_performance.py --mode all --show  # All tests with visualizations
  python test_yolo_performance.py -m cache filter --show # Cache and filter tests with viz
        """
    )
    
    parser.add_argument(
        "--mode", "-m",
        nargs="+",
        choices=["single", "batch", "cache", "filter", "compare", "all"],
        default=["single"],
        help="Test mode(s) to run (default: single)"
    )
    
    parser.add_argument(
        "--show", "-s",
        action="store_true",
        help="Create and save beautiful visualizations of detection results (saved to outputs/ folder)"
    )
    
    args = parser.parse_args()
    
    # Discover test images
    original_images = get_test_images()
    
    if not original_images:
        print("❌ No images found in 'images' folder!")
        print("   Place some test images (.jpg, .png, etc.) in the images/ directory")
        return
    
    print(f"🚀 Testing YOLO Optimizations with {len(original_images)} images")
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
        modes = ["single", "batch", "cache", "filter", "compare"]
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
                detector = YOLODetector(YOLOConfig(enable_timing=True))
            test_model_cache(detector, original_images, args.show)
        elif mode == "filter":
            test_content_filtering(original_images, args.show)
        elif mode == "compare":
            test_batch_vs_individual(original_images, args.show)
    
    if args.show:
        print(f"\n✅ All visualizations saved to: outputs/ folder")
        print(f"🔍 Check the outputs/ directory for beautiful YOLO detection visualizations!")

if __name__ == "__main__":
    main()


# Usage Examples:

# # Default: Single image test only (no visualization)
# uv run tests/test_yolo_performance.py

# # Single test with beautiful visualization
# uv run tests/test_yolo_performance.py --show

# # Specific modes with visualization
# uv run tests/test_yolo_performance.py --mode batch --show
# uv run tests/test_yolo_performance.py --mode filter --show
# uv run tests/test_yolo_performance.py --mode compare --show

# # Multiple modes with visualization
# uv run tests/test_yolo_performance.py -m single cache filter --show
# uv run tests/test_yolo_performance.py -m batch compare --show

# # All tests with beautiful visualizations
# uv run tests/test_yolo_performance.py --mode all --show

# # Help
# uv run tests/test_yolo_performance.py --help