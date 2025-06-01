import sys
import os
import glob

# Add parent directory to path so we can import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psutil
from utils.yolo_detector import YOLODetector, YOLOConfig

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

def test_memory_usage():
    """Test memory efficiency (YOLOMemoryPool removal)"""
    
    # Find test images
    test_images = get_test_images()
    
    if not test_images:
        print("❌ No images found in 'images' folder!")
        print("   Place some test images (.jpg, .png, etc.) in the images/ directory")
        return
    
    print(f"🧠 Memory Usage Test with {len(test_images)} images")
    print(f"📁 Using images: {[os.path.basename(img) for img in test_images[:3]]}")
    if len(test_images) > 3:
        print(f"   ... and {len(test_images) - 3} more")
    
    process = psutil.Process(os.getpid())
    
    # Before creating detector
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    config = YOLOConfig(enable_timing=False)  # Disable timing for cleaner output
    detector = YOLODetector(config)
    
    # After creating detector
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    
    # Process first few images individually
    test_subset = test_images[:min(3, len(test_images))]
    
    print(f"\n🔄 Processing {len(test_subset)} images individually...")
    for i, img_path in enumerate(test_subset):
        print(f"  Processing {i+1}: {os.path.basename(img_path)}")
        detections = detector.detect(img_path)
        print(f"    → {len(detections)} detections")
    
    # After processing
    mem_final = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"\n📊 Memory Usage Results:")
    print(f"   🏁 Before initialization: {mem_before:.1f} MB")
    print(f"   🚀 After detector init:   {mem_after:.1f} MB (+{mem_after-mem_before:.1f} MB)")
    print(f"   🎯 After processing:      {mem_final:.1f} MB (+{mem_final-mem_after:.1f} MB)")
    print(f"   📈 Total memory increase:  {mem_final-mem_before:.1f} MB")
    
    # Test model cache reset memory impact
    print(f"\n🔄 Testing model cache reset...")
    mem_before_reset = process.memory_info().rss / 1024 / 1024
    detector.reset_model_cache()
    mem_after_reset = process.memory_info().rss / 1024 / 1024
    
    print(f"   Memory before reset: {mem_before_reset:.1f} MB")
    print(f"   Memory after reset:  {mem_after_reset:.1f} MB")
    if mem_after_reset < mem_before_reset:
        print(f"   ✅ Memory freed:        {mem_before_reset-mem_after_reset:.1f} MB")
    else:
        print(f"   ⚠️  Memory increased:    +{mem_after_reset-mem_before_reset:.1f} MB")
    
    # Test multiple detectors (to check for memory leaks)
    print(f"\n🔍 Testing multiple detector instances...")
    mem_before_multi = process.memory_info().rss / 1024 / 1024
    
    detectors = []
    for i in range(3):
        det = YOLODetector(YOLOConfig(enable_timing=False))
        detectors.append(det)
        det.detect(test_subset[0])  # Process one image with each
    
    mem_after_multi = process.memory_info().rss / 1024 / 1024
    print(f"   Memory before multi-test: {mem_before_multi:.1f} MB")
    print(f"   Memory after 3 detectors: {mem_after_multi:.1f} MB")
    print(f"   Memory per detector:      {(mem_after_multi-mem_before_multi)/3:.1f} MB")
    
    # Clean up
    del detectors
    
    print(f"\n✅ Memory test completed!")

if __name__ == "__main__":
    test_memory_usage()
