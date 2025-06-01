"""
Pipeline V1: Configuration Setup + YOLO/OCR Detection + Merging + Seraphine Grouping + Visualizations
Building the complete pipeline step by step - with intelligent ID tracking and JSON export
Set mode = "debug" for prints and files to work, else deploy_mcp
"""
import os
import time
import json
import cv2
import numpy as np
from functools import wraps
from PIL import Image
from datetime import datetime
from utils.yolo_detector import YOLODetector, YOLOConfig
from utils.ocr_detector import OCRDetector, OCRDetConfig
from utils.bbox_merger import BBoxMerger
from utils.beautiful_visualizer import BeautifulVisualizer
from utils.seraphine_processor import FinalSeraphineProcessor, convert_detections_to_seraphine_format
from utils.seraphine_generator import FinalGroupImageGenerator
import asyncio
from utils.gemini_integration import run_gemini_analysis, integrate_gemini_results
from utils.pipeline_exporter import save_enhanced_pipeline_json
from concurrent.futures import ThreadPoolExecutor
from utils.parallel_processor import ParallelProcessor
from utils.helpers import load_configuration, debug_print




def setup_detector_configs(config):
    """Setup YOLO and OCR configurations from config.json"""
    
    # Configure YOLO from config.json
    yolo_config = YOLOConfig(
        model_path=config.get("yolo_model_path", "models/model_dynamic.onnx"),
        conf_threshold=config.get("yolo_conf_threshold", 0.1),
        iou_threshold=config.get("yolo_iou_threshold", 0.1),
        enable_timing=config.get("yolo_enable_timing", True),
        enable_debug=config.get("yolo_enable_debug", False)
    )
    
    # Configure OCR from config.json
    ocr_config = OCRDetConfig(
        model_path=config.get("ocr_model_path", "models/ch_PP-OCRv3_det_infer.onnx"),
        det_threshold=config.get("ocr_det_threshold", 0.3),
        max_side_len=config.get("ocr_max_side_len", 960),
        enable_timing=config.get("ocr_enable_timing", True),
        enable_debug=config.get("ocr_enable_debug", False),
        use_dilation=config.get("ocr_use_dilation", True)
    )
    
    debug_print(f"🎯 YOLO Config: conf={yolo_config.conf_threshold}, iou={yolo_config.iou_threshold}")
    debug_print(f"📝 OCR Config: threshold={ocr_config.det_threshold}, max_len={ocr_config.max_side_len}")
    
    return yolo_config, ocr_config

def load_image_opencv(image_path):
    """Load image using OpenCV (no PIL)"""
    if not os.path.exists(image_path):
        debug_print(f"❌ Error: Image file '{image_path}' not found!")
        return None
    
    # Load with OpenCV
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        debug_print(f"❌ Error: Could not load image '{image_path}'")
        return None
    
    debug_print(f"📸 Image loaded: {img_bgr.shape[1]}x{img_bgr.shape[0]} pixels")
    return img_bgr

def convert_bgr_to_pil_for_ocr(img_bgr):
    """Convert OpenCV BGR to PIL RGB for OCR detector (temporary until we optimize OCR)"""
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Convert to PIL (OCR still needs PIL for now)
    pil_image = Image.fromarray(img_rgb)
    return pil_image

def assign_intelligent_ids(yolo_detections, ocr_detections):
    """
    Assign clean, simple IDs for tracking throughout pipeline
    YOLO: y_id only (remove redundant 'id')
    OCR: o_id only (remove redundant 'id')
    """
    debug_print("🔖 Assigning simple, clean IDs for pipeline tracking...")
    
    # Assign YOLO IDs - CLEAN VERSION (remove 'id' field)
    for i, detection in enumerate(yolo_detections):
        detection['y_id'] = f"Y{i+1:03d}"
        # Remove redundant 'id' field completely
        if 'id' in detection:
            del detection['id']
    
    # Assign OCR IDs - CLEAN VERSION (remove 'id' field)
    for i, detection in enumerate(ocr_detections):
        detection['o_id'] = f"O{i+1:03d}"
        # Remove redundant 'id' field completely  
        if 'id' in detection:
            del detection['id']
    
    debug_print(f"  ✅ Assigned {len(yolo_detections)} YOLO IDs (Y001-Y{len(yolo_detections):03d})")
    debug_print(f"  ✅ Assigned {len(ocr_detections)} OCR IDs (O001-O{len(ocr_detections):03d})")
    
    return yolo_detections, ocr_detections

def create_intelligent_merger(config):
    """Create enhanced merger with CLEAN ID tracking and proper field ordering"""
    
    class IntelligentBBoxMerger(BBoxMerger):
        """Enhanced merger with clean ID tracking and proper field ordering"""
        
        def merge_detections(self, yolo_detections, ocr_detections):
            """Enhanced merge with CLEAN ID tracking and m_id first"""
            import time
            start_time = time.time()
            
            # Call parent merge method
            merged_detections, merge_stats = super().merge_detections(yolo_detections, ocr_detections)
            
            # Create new, properly ordered detections
            cleaned_merged = []
            
            for i, detection in enumerate(merged_detections):
                # Create new detection with proper field order: m_id, y_id, o_id
                cleaned_detection = {}
                
                # 1. Primary ID first (most important)
                cleaned_detection['m_id'] = f"M{i+1:03d}"
                
                # 2. Source tracking (in logical order)
                if detection['source'] == 'yolo':
                    # Came from YOLO - has y_id, no o_id
                    cleaned_detection['y_id'] = detection.get('y_id', f"Y{i+1:03d}")
                    cleaned_detection['o_id'] = "NA"
                elif detection['source'] == 'ocr_det':
                    # Came from OCR - has o_id, no y_id  
                    cleaned_detection['y_id'] = "NA"
                    cleaned_detection['o_id'] = detection.get('o_id', f"O{i+1:03d}")
                else:
                    # Shouldn't happen, but future-proofing
                    cleaned_detection['y_id'] = "NA"
                    cleaned_detection['o_id'] = "NA"
                
                # 3. Copy other important fields (in original order)
                for key in ['bbox', 'type', 'source', 'confidence']:
                    if key in detection:
                        cleaned_detection[key] = detection[key]
                
                # Add any other fields except the ones we want to remove
                excluded_fields = {'id', 'merged_id', 'merge_id', 'source_ids', 'relationship', 
                                 'original_id', 'yolo_id', 'ocr_id', 'y_id', 'o_id', 'm_id',
                                 'bbox', 'type', 'source', 'confidence'}
                
                for key, value in detection.items():
                    if key not in excluded_fields:
                        cleaned_detection[key] = value
                
                cleaned_merged.append(cleaned_detection)
            
            return cleaned_merged, merge_stats
    
    return IntelligentBBoxMerger(
        iou_threshold=config.get("merger_iou_threshold", 0.05),
        containment_threshold=0.8,
        min_area=1.0,
        enable_timing=True
    )

def run_parallel_detection_and_merge(img_bgr, yolo_config, ocr_config, config):
    """
    Step 1: Run YOLO + OCR detection + intelligent merging (FIXED - using ParallelProcessor!)
    """
    debug_print("\n🔄 Step 1: Parallel YOLO + OCR Detection + Intelligent Merging (FIXED)")
    debug_print("=" * 60)
    
    # 🎯 FIXED: Use ParallelProcessor properly (like temp_main.py)
    parallel_processor = ParallelProcessor(
        yolo_config=yolo_config,
        ocr_config=ocr_config,
        merger_iou_threshold=config.get("merger_iou_threshold"),
        enable_timing=config.get("yolo_enable_timing", True),
        create_visualizations=False,  # We handle visualizations separately
        save_intermediate_results=False  # We handle JSON separately
    )
    
    # For now, save BGR as temp file for ParallelProcessor 
    # (until ParallelProcessor supports direct PIL/numpy input)
    temp_image_path = "temp_detection_image.jpg"
    cv2.imwrite(temp_image_path, img_bgr)
    
    try:
        detection_start = time.time()
        
        # 🎯 Use the PROPER ParallelProcessor with full merging logic!
        results = parallel_processor.process_image(temp_image_path, "temp")
        
        total_detection_time = time.time() - detection_start
        
        # Extract results (ParallelProcessor returns proper structure)
        yolo_detections = results['yolo_detections']
        ocr_detections = results['ocr_detections'] 
        merged_detections = results['merged_detections']
        merge_stats = results['merge_stats']
        
        # Assign intelligent IDs for tracking (same as before)
        yolo_detections, ocr_detections = assign_intelligent_ids(yolo_detections, ocr_detections)
        
        # Update merged detections with proper IDs
        for i, detection in enumerate(merged_detections):
            detection['m_id'] = f"M{i+1:03d}"
        
        debug_print(f"\n📊 FIXED Detection + Merge Results:")
        debug_print(f"  🎯 YOLO detections: {len(yolo_detections)} (Y001-Y{len(yolo_detections):03d})")
        debug_print(f"  📝 OCR detections: {len(ocr_detections)} (O001-O{len(ocr_detections):03d})")
        debug_print(f"  🔗 MERGED detections: {len(merged_detections)} (M001-M{len(merged_detections):03d})")
        debug_print(f"  ⏱️  Total time: {total_detection_time:.3f}s")
        debug_print(f"  🎯 PROPER 3-stage merging logic restored!")
        debug_print(f"  📈 Merge efficiency: {len(yolo_detections) + len(ocr_detections)} → {len(merged_detections)} ({len(yolo_detections) + len(ocr_detections) - len(merged_detections)} removed)")
        
        return {
            'yolo_detections': yolo_detections,
            'ocr_detections': ocr_detections, 
            'merged_detections': merged_detections,
            'merge_stats': merge_stats,
            'timing': {
                'total_detection_time': total_detection_time,
                'parallel_detection_time': results['timing']['parallel_detection_time'],
                'merge_time': results['timing']['merge_time']
            }
        }
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

def run_seraphine_grouping(merged_detections, config):
    """
    Step 2: Run Seraphine intelligent grouping with perfect m_id tracking
    """
    debug_print("\n🧠 Step 2: Seraphine Intelligent Grouping & Layout Analysis")
    debug_print("=" * 60)
    
    if not merged_detections:
        debug_print("⚠️  No merged detections to group!")
        return None
    
    seraphine_start = time.time()
    
    # Convert merged detections to seraphine format with PERFECT m_id mapping
    seraphine_detections = convert_merged_to_seraphine_format(merged_detections)
    
    # Initialize seraphine processor
    seraphine_processor = FinalSeraphineProcessor(
        enable_timing=config.get("seraphine_enable_timing", True),
        enable_debug=config.get("seraphine_enable_debug", False)
    )
    
    # Process detections into groups
    seraphine_analysis = seraphine_processor.process_detections(seraphine_detections)
    
    # Create enhanced analysis with perfect m_id → group tracking
    enhanced_analysis = create_seraphine_id_mapping(seraphine_analysis, merged_detections)
    
    seraphine_time = time.time() - seraphine_start
    
    # FIXED: Access nested analysis values correctly
    analysis = enhanced_analysis['analysis']
    
    # Results summary  
    debug_print(f"\n📊 Seraphine Grouping Results:")
    debug_print(f"  🧠 Input: {len(merged_detections)} merged detections (M001-M{len(merged_detections):03d})")
    debug_print(f"  📦 Groups created: {analysis['total_groups']}")
    debug_print(f"  📐 Horizontal groups: {analysis['horizontal_groups']}")
    debug_print(f"  📏 Vertical groups: {analysis['vertical_groups']}")
    debug_print(f"  📏 Long box groups: {analysis['long_box_groups']}")
    debug_print(f"  ⏱️  Seraphine time: {seraphine_time:.3f}s")
    debug_print(f"  🔗 Perfect m_id tracking: M001 → H1_1, M002 → H1_2, etc.")
    
    # Add timing info to enhanced analysis
    enhanced_analysis['seraphine_timing'] = seraphine_time
    
    return enhanced_analysis

def convert_merged_to_seraphine_format(merged_detections):
    """
    Convert merged detections to seraphine format with PERFECT m_id preservation
    This is the CRITICAL step where m_id tracking must not break!
    """
    debug_print("🔗 Converting merged detections to seraphine format (preserving m_ids)...")
    
    seraphine_detections = []
    
    for detection in merged_detections:
        # CRITICAL: Use m_id as the primary ID for seraphine
        m_id = detection['m_id']  # e.g., "M001"
        
        seraphine_detection = {
            'bbox': detection['bbox'],
            'id': m_id,  # CRITICAL: This will be used by seraphine for group mapping
            'merged_id': m_id,  # Keep reference
            'type': detection.get('type', 'unknown'),
            'source': detection.get('source', 'merged'),
            'confidence': detection.get('confidence', 1.0),
            # Keep original tracking info for reference
            'y_id': detection.get('y_id', 'NA'),
            'o_id': detection.get('o_id', 'NA')
        }
        
        seraphine_detections.append(seraphine_detection)
    
    debug_print(f"  ✅ Converted {len(seraphine_detections)} detections with preserved m_ids")
    return seraphine_detections

def create_seraphine_id_mapping(seraphine_analysis, merged_detections):
    """
    Create enhanced analysis with perfect m_id → seraphine group tracking
    """
    debug_print("🗺️  Creating enhanced m_id → seraphine group mapping...")
    
    # Get the bbox processor from seraphine analysis  
    bbox_processor = seraphine_analysis['bbox_processor']
    
    # Get the mapping: m_id → group_label (e.g., "M001" → "H1_1")
    m_id_to_group = bbox_processor.bbox_to_group_mapping
    
    # Create reverse mapping for easy lookup
    group_to_m_ids = {}
    for m_id, group_label in m_id_to_group.items():
        if group_label not in group_to_m_ids:
            group_to_m_ids[group_label] = []
        group_to_m_ids[group_label].append(m_id)
    
    # Enhance the original analysis with tracking info
    enhanced_analysis = seraphine_analysis.copy()
    enhanced_analysis.update({
        'm_id_to_group_mapping': m_id_to_group,  # "M001" → "H1_1"
        'group_to_m_ids_mapping': group_to_m_ids,  # "H1_1" → ["M001", "M002"]
        'total_m_ids_grouped': len(m_id_to_group),
        'seraphine_timing': seraphine_analysis.get('processing_time', 0)
    })
    
    debug_print(f"  ✅ Enhanced mapping created:")
    debug_print(f"     📦 {len(group_to_m_ids)} groups with m_id tracking")
    debug_print(f"     🔗 {len(m_id_to_group)} m_ids mapped to groups")
    
    # debug_print sample mappings for verification
    debug_print(f"  📋 Sample m_id → group mappings:")
    for i, (m_id, group_label) in enumerate(list(m_id_to_group.items())[:5]):
        debug_print(f"     {m_id} → {group_label}")
    if len(m_id_to_group) > 5:
        debug_print(f"     ... and {len(m_id_to_group) - 5} more")
    
    return enhanced_analysis

def create_visualizations(image_path, detection_results, seraphine_analysis, config, gemini_results=None):
    """
    Step 6: Create beautiful visualizations (respecting config settings)
    """
    if not config.get("save_visualizations", False):
        debug_print("\n⏭️  Visualizations disabled in config (save_visualizations: false)")
        return None
    
    debug_print("\n🎨 Step 6: Creating Enhanced Visualizations (with Seraphine Groups)")
    debug_print("=" * 70)
    
    import time
    from utils.beautiful_visualizer import BeautifulVisualizer
    
    output_dir = config.get("output_dir", "outputs")
    filename_base = os.path.splitext(os.path.basename(image_path))[0]
    
    viz_start = time.time()
    
    # Initialize visualizer with config
    visualizer = BeautifulVisualizer(output_dir=output_dir, config=config)
    
    # Create traditional visualizations (respecting config)
    viz_results = {
        'yolo_detections': detection_results['yolo_detections'],     # Blue boxes
        'ocr_detections': detection_results['ocr_detections'],       # Green boxes
        'merged_detections': detection_results['merged_detections']  # Purple boxes (intelligently merged)
    }
    
    # Create traditional visualizations using existing method
    visualization_paths = visualizer.create_all_visualizations(
        image_path=image_path,
        results=viz_results,
        filename_base=f"v1_{filename_base}"
    )
    
    # Create seraphine group visualization using existing method
    if seraphine_analysis and config.get("save_seraphine_viz", True):
        seraphine_path = visualizer.create_seraphine_group_visualization(
            image_path=image_path,
            seraphine_analysis=seraphine_analysis,
            filename_base=f"v1_{filename_base}"
        )
        if seraphine_path:
            visualization_paths['seraphine_groups'] = seraphine_path
    
    # 🆕 Create Gemini visualization using correct format
    if gemini_results and config.get("save_gemini_visualization", True):
        debug_print("🎨 Creating Gemini analysis visualization...")
        try:
            from PIL import Image as PILImage
            original_image = PILImage.open(image_path)
            
            # Use the EXACT gemini_results format - the visualizer expects this!
            gemini_viz_path = visualizer._create_gemini_visualization(
                image=original_image,
                gemini_analysis=gemini_results,  # Pass the full results object!
                seraphine_analysis=seraphine_analysis,
                filename_base=f"v1_{filename_base}"
            )
            
            if gemini_viz_path:
                visualization_paths['gemini_analysis'] = gemini_viz_path
                debug_print(f"✅ Gemini visualization created: {os.path.basename(gemini_viz_path)}")
        except Exception as e:
            debug_print(f"⚠️  Failed to create Gemini visualization: {e}")
            import traceback
            traceback.print_exc()
    
    viz_time = time.time() - viz_start
    
    debug_print(f"✅ Enhanced visualizations created in {viz_time:.3f}s:")
    for viz_type, path in visualization_paths.items():
        if isinstance(path, str):
            debug_print(f"   📷 {viz_type.upper()}: {os.path.basename(path)}")
        else:
            debug_print(f"   📷 {viz_type.upper()}: {len(path)} files")
    
    return visualization_paths

def display_enhanced_pipeline_summary(image_path, detection_results, seraphine_analysis, gemini_results, visualization_paths, json_path, config):
    """Display enhanced pipeline summary with seraphine integration"""
    merge_stats = detection_results['merge_stats']
    
    # FIXED: Access nested analysis values correctly
    analysis = seraphine_analysis['analysis']
    
    debug_print(f"\n📊 ENHANCED PIPELINE V1.2 SUMMARY (with Seraphine and Gemini):")
    debug_print("=" * 65)
    debug_print(f"  📸 Image: {os.path.basename(image_path)}")
    debug_print(f"  🎯 YOLO detections: {len(detection_results['yolo_detections'])} (Y001-Y{len(detection_results['yolo_detections']):03d})")
    debug_print(f"  📝 OCR detections: {len(detection_results['ocr_detections'])} (O001-O{len(detection_results['ocr_detections']):03d})")
    debug_print(f"  🔗 MERGED detections: {len(detection_results['merged_detections'])} (M001-M{len(detection_results['merged_detections']):03d})")
    debug_print(f"  🧠 SERAPHINE groups: {analysis['total_groups']} intelligent groups")
    debug_print(f"     📐 Horizontal: {analysis['horizontal_groups']}")
    debug_print(f"     📏 Vertical: {analysis['vertical_groups']}")
    debug_print(f"     📏 Long boxes: {analysis['long_box_groups']}")
    
    # Handle None gemini_results properly
    gemini_time = gemini_results.get('analysis_duration_seconds', 0) if gemini_results else 0
    total_time = detection_results['timing']['total_detection_time'] + seraphine_analysis.get('seraphine_timing', 0) + gemini_time
    
    debug_print(f"  ⏱️  Total pipeline time: {total_time:.3f}s")
    debug_print(f"     Detection + merge: {detection_results['timing']['total_detection_time']:.3f}s")
    debug_print(f"     Seraphine grouping: {seraphine_analysis.get('seraphine_timing', 0):.3f}s")
    debug_print(f"     Gemini analysis: {gemini_time:.3f}s")
    
    # Show Gemini status
    if gemini_results:
        debug_print(f"  🤖 GEMINI analysis: ✅ {gemini_results.get('successful_analyses', 0)}/{gemini_results.get('total_images_analyzed', 0)} images analyzed")
        debug_print(f"     🎯 Total icons found: {gemini_results.get('total_icons_found', 0)}")
    else:
        debug_print(f"  🤖 GEMINI analysis: ⏭️ Disabled or failed")
    
    if json_path:
        debug_print(f"  💾 Enhanced JSON: {os.path.basename(json_path)}")
        debug_print(f"     - Complete pipeline with seraphine group tracking and Gemini analysis")
        debug_print(f"     - Perfect m_id → seraphine_group mapping")
        debug_print(f"     - {seraphine_analysis['total_m_ids_grouped']} m_ids tracked through {analysis['total_groups']} groups")
    
    if visualization_paths:
        debug_print(f"  🎨 Visualizations: {len(visualization_paths)} types created")
        debug_print(f"     - Traditional: YOLO, OCR, MERGED overlays")
        if 'seraphine_groups' in visualization_paths:
            seraphine_count = len(visualization_paths['seraphine_groups']) if isinstance(visualization_paths['seraphine_groups'], list) else 1
            debug_print(f"     - Seraphine: {seraphine_count} intelligent group images")
        debug_print(f"  📁 Output directory: {config.get('output_dir', 'outputs')}/")

    debug_print(f"🔗 Perfect ID Traceability: Y/O IDs → M IDs → Seraphine Groups → Gemini Analysis")

async def main():
    """Main enhanced pipeline execution - MODE AWARE"""
    pipeline_start = time.time()
    
    config = load_configuration()
    if not config:
        return None
    
    mode = config.get("mode", "debug")
    
    # Force disable ALL debug output in deploy mode
    if mode == "deploy_mcp":
        config.update({
            "yolo_enable_debug": False,
            "yolo_enable_timing": False,
            "ocr_enable_debug": False,
            "ocr_enable_timing": False,
            "seraphine_enable_debug": False,
            "seraphine_timing": False,
            "save_visualizations": False,
            "save_json": False,
            "save_gemini_visualization": False,
            "save_gemini_json": False,
        })
    
    debug_print("🚀 ENHANCED AI PIPELINE V1.2: Detection + Merging + Seraphine + Gemini + Export")
    debug_print("=" * 90)
    
    yolo_config, ocr_config = setup_detector_configs(config)
    image_path = "images/word.png"
    img_bgr = load_image_opencv(image_path)
    if img_bgr is None:
        return None
    
    debug_print(f"📸 Image loaded: {img_bgr.shape[1]}x{img_bgr.shape[0]} pixels")
    
    try:
        # Step 1: Detection + Merging
        detection_results = run_parallel_detection_and_merge(img_bgr, yolo_config, ocr_config, config)
        
        # Step 2: Seraphine Grouping
        seraphine_analysis = run_seraphine_grouping(detection_results['merged_detections'], config)
        
        # Step 3: Generate Grouped Images
        grouped_image_paths = None
        if config.get("generate_grouped_images", True):
            debug_print("\n🖼️  Step 3: Generating Seraphine Grouped Images")
            
            from utils.seraphine_generator import FinalGroupImageGenerator
            
            output_dir = config.get("output_dir", "outputs")
            filename_base = os.path.splitext(os.path.basename(image_path))[0]
            
            final_group_generator = FinalGroupImageGenerator(
                output_dir=output_dir,
                save_mapping=False
            )
            
            grouped_image_paths = final_group_generator.create_grouped_images(
                image_path, 
                seraphine_analysis, 
                filename_base
            )
            
            debug_print(f"✅ Generated {len(grouped_image_paths)} grouped images")
        
        # Step 4: Gemini Analysis
        gemini_results = None
        if config.get("gemini_enabled", False):
            try:
                gemini_results = await run_gemini_analysis(
                    seraphine_analysis, grouped_image_paths, image_path, config
                )
                
                if gemini_results:
                    seraphine_analysis = integrate_gemini_results(seraphine_analysis, gemini_results)
                    
            except Exception as e:
                debug_print(f"⚠️  Gemini analysis failed: {str(e)}")
        
        # Calculate total time BEFORE mode check
        total_time = time.time() - pipeline_start
        
        # Get icon count BEFORE mode check
        icon_count = gemini_results.get('total_icons_found', 0) if gemini_results else 0
        
        # MODE-SPECIFIC OUTPUTS
        if mode == "deploy_mcp":
            # 🎯 DEPLOY MODE: Clean, emoji-free output
            print(f"Pipeline completed in {total_time:.3f}s, found {icon_count} icons.")
            
            # 🧹 COMPLETE FILE CLEANUP
            output_dir = config.get("output_dir", "outputs")
            if os.path.exists(output_dir):
                import shutil
                shutil.rmtree(output_dir)
                os.makedirs(output_dir, exist_ok=True)
            
            # Return only essential data
            field_name = 'seraphine_gemini_groups' if gemini_results else 'seraphine_groups'
            return {
                'total_time': total_time,
                'total_icons_found': icon_count,
                field_name: seraphine_analysis.get(field_name, seraphine_analysis.get('analysis', {}))
            }
        
        else:  # DEBUG MODE - Full verbose output with emojis
            # Step 5: Save JSON
            json_path = save_enhanced_pipeline_json(image_path, detection_results, seraphine_analysis, gemini_results, config)
            
            # Step 6: Create Visualizations
            visualization_paths = create_visualizations(image_path, detection_results, seraphine_analysis, config, gemini_results)
            
            # Summary
            display_enhanced_pipeline_summary(image_path, detection_results, seraphine_analysis, gemini_results, visualization_paths, json_path, config)
            
            return {
                'detection_results': detection_results,
                'seraphine_analysis': seraphine_analysis,
                'gemini_results': gemini_results,
                'grouped_image_paths': grouped_image_paths,
                'visualization_paths': visualization_paths,
                'json_path': json_path,
                'config': config,
                'total_time': total_time
            }
        
    except Exception as e:
        # Calculate time even on error
        total_time = time.time() - pipeline_start
        
        if mode == "deploy_mcp":
            # Clean error message without emojis
            print(f"Pipeline failed after {total_time:.3f}s: {str(e)}")
        else:
            debug_print(f"❌ Error during pipeline execution: {str(e)}")
            import traceback
            traceback.print_exc()  # Fixed: remove debug_print_exc
        
        return None

if __name__ == "__main__":
    try:
        results = asyncio.run(main())
    except Exception as e:
        print(f"Critical startup error: {e}")