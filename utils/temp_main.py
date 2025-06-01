"""
Complete pipeline with Final Seraphine integration, beautiful visualizations, grouped images, and Gemini LLM analysis
"""
import os
import time
import json
import asyncio
from PIL import Image
from utils.yolo_detector import YOLODetector, YOLOConfig
from utils.ocr_detector import OCRDetector, OCRDetConfig  
from utils.bbox_merger import BBoxMerger
from utils.seraphine_processor import FinalSeraphineProcessor, convert_detections_to_seraphine_format
from utils.seraphine_generator import FinalGroupImageGenerator
from utils.beautiful_visualizer import BeautifulVisualizer
from utils.gemini_analyzer import GeminiIconAnalyzer
from utils.parallel_processor import ParallelProcessor

with open("utils/config.json", "r") as f:
    config = json.load(f)

async def main():
    """Complete pipeline with Final Seraphine integration, visualizations, and Gemini LLM analysis"""
    print("🚀 COMPLETE AI PIPELINE: YOLO + OCR + Final Seraphine + Visualizations + Gemini LLM")
    print("=" * 90)
    
    # Configuration
    image_path = "altair.jpg"
    output_dir = config.get("output_dir", "outputs")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file '{image_path}' not found!")
        return
    
    # Configure components based on config.json
    yolo_config = YOLOConfig(
        model_path=config.get("yolo_model_path"),
        conf_threshold=config.get("yolo_conf_threshold"),
        iou_threshold=config.get("yolo_iou_threshold"),
        enable_timing=config.get("yolo_enable_timing"),
        enable_debug=config.get("yolo_enable_debug")
    )
    
    ocr_config = OCRDetConfig(
        model_path=config.get("ocr_model_path"),
        det_threshold=config.get("ocr_det_threshold"),
        max_side_len=config.get("ocr_max_side_len"),
        enable_timing=config.get("ocr_enable_timing"),
        enable_debug=config.get("ocr_enable_debug"),
        use_dilation=config.get("ocr_use_dilation")
    )
    
    # Initialize processors
    parallel_processor = ParallelProcessor(
        yolo_config=yolo_config,
        ocr_config=ocr_config,
        merger_iou_threshold=config.get("merger_iou_threshold"),
        enable_timing=config.get("seraphine_timing", True),
        create_visualizations=False,
        save_intermediate_results=config.get("save_json", False)
    )
    
    # Initialize Seraphine components
    final_seraphine_processor = FinalSeraphineProcessor(
        enable_timing=config.get("seraphine_timing"), 
        enable_debug=config.get("seraphine_enable_debug")
    )
    
    final_group_generator = FinalGroupImageGenerator(
        output_dir=output_dir,
        save_mapping=config.get("save_json", False)
    )
    
    # Initialize Gemini analyzer (optional - will skip if not configured)
    gemini_analyzer = None
    try:
        # gemini_analyzer = None
        gemini_analyzer = GeminiIconAnalyzer(
            prompt_path=config.get("gemini_prompt_path"), 
            output_dir=output_dir,
            save_results=config.get("save_json", False)  # NEW: Pass config setting
        )
        if not config.get("yolo_enable_debug"):  # Only print if not in debug mode
            print("✅ Gemini analyzer initialized")
    except Exception as e:
        print(f"⚠️  Gemini analyzer not available: {e}")
        print("   Continuing without LLM analysis...")
    
    try:
        # Step 1: Run parallel YOLO + OCR detection
        if config.get("yolo_enable_debug") or config.get("ocr_enable_debug"):
            print("\n🔄 Step 1: Parallel YOLO + OCR Detection")
        
        results = parallel_processor.process_image(image_path, output_dir)
        
        # Step 2: Run Final Seraphine analysis
        if config.get("seraphine_enable_debug"):
            print("\n🧠 Step 2: Final Seraphine Intelligent Grouping & Image Generation")
        
        seraphine_detections = convert_detections_to_seraphine_format(results['merged_detections'])
        seraphine_analysis = final_seraphine_processor.process_detections(seraphine_detections)
        results['seraphine_analysis'] = seraphine_analysis
        
        # Step 3: Generate grouped images for Gemini (only if Gemini is available)
        filename_base = os.path.splitext(os.path.basename(image_path))[0]
        
        if gemini_analyzer:
            if config.get("seraphine_enable_debug"):
                print("\n🖼️  Step 3: Creating Grouped Images for Gemini Analysis")
            
            # Use direct images mode for better performance
            grouped_results = final_group_generator.create_grouped_images(
                image_path, 
                seraphine_analysis, 
                filename_base, 
                return_direct_images=config.get("gemini_return_images_b64", False)
            )
            
            # Handle both modes properly
            if config.get("gemini_return_images_b64", False):
                # Dictionary mode with direct images
                results['grouped_image_paths'] = grouped_results.get('file_paths', [])
                results['direct_images'] = grouped_results.get('direct_images', [])
                image_count = grouped_results.get('image_count', 0)
            else:
                # Traditional file mode - returns list directly
                results['grouped_image_paths'] = grouped_results if isinstance(grouped_results, list) else []
                results['direct_images'] = None
                image_count = len(results['grouped_image_paths'])
            
            if config.get("seraphine_enable_debug"):
                print(f"✅ Generated {image_count} grouped images for analysis")
        
        
        # Step 4: Gemini LLM Analysis
        if gemini_analyzer:
            if config.get("seraphine_enable_debug"):
                print("\n🤖 Step 4: Gemini LLM Analysis of Grouped Icons")
            
            try:
                # Use direct images if available, otherwise use file paths
                if results.get('direct_images'):
                    gemini_results = await gemini_analyzer.analyze_grouped_images(
                        grouped_image_paths=None,
                        filename_base=filename_base,
                        direct_images=results['direct_images']
                    )
                else:
                    gemini_results = await gemini_analyzer.analyze_grouped_images(
                        grouped_image_paths=results.get('grouped_image_paths', []),
                        filename_base=filename_base,
                        direct_images=None
                    )
                
                results['gemini_analysis'] = gemini_results
                
                if config.get("seraphine_enable_debug"):
                    print(f"✅ Gemini analysis completed: {gemini_results['total_icons_found']} icons identified")
                
            except Exception as e:
                print(f"❌ Gemini analysis failed: {e}")
                results['gemini_analysis'] = None
        
        # Step 5: Save final results JSON (only if enabled in config)
        if config.get("save_json", False):
            if config.get("seraphine_enable_debug"):
                print("\n💾 Step 5: Saving Final Results")
            save_final_results_json(results, output_dir, filename_base)
        
        # Step 6: Create visualizations (only if enabled in config)
        if config.get("save_visualizations", False):
            if config.get("seraphine_enable_debug"):
                print("\n🎨 Step 6: Creating Beautiful Visualizations")
            visualizer = BeautifulVisualizer(output_dir=output_dir)
            visualization_paths = visualizer.create_all_visualizations(
                image_path, results, filename_base
            )
            results['visualization_paths'] = visualization_paths
        
        # Step 7: Clean up temporary files (if save_images is False)
        if not config.get("save_images", False) and results.get('grouped_image_paths'):
            cleanup_temp_files(results.get('grouped_image_paths', []))
        
        # Step 8: Optional Gemini visualization and JSON export
        if results.get('gemini_analysis') and (config.get("save_gemini_visualization", False) or config.get("save_gemini_json", False)):
            if config.get("seraphine_enable_debug"):
                print("\n🎨 Step 8: Creating Gemini Visualization and Saving Results")
            
            if config.get("save_gemini_visualization", False):
                # Create gemini visualization
                visualizer = BeautifulVisualizer(output_dir=output_dir)
                original_image = Image.open(image_path)
                gemini_viz_path = visualizer._create_gemini_visualization(
                    original_image, 
                    results['gemini_analysis'], 
                    results['seraphine_analysis'], 
                    filename_base
                )
                results['gemini_visualization_path'] = gemini_viz_path
            
            if config.get("save_gemini_json", False):
                # Save gemini results as separate JSON
                save_gemini_results_json(results.get('gemini_analysis'), output_dir, filename_base)
        
        # Final Summary (always show)
        display_final_summary(results, image_path, output_dir)

        return results
        
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def cleanup_temp_files(file_paths: list):
    """Clean up temporary grouped image files"""
    cleaned_count = 0
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                cleaned_count += 1
        except Exception as e:
            print(f"⚠️  Could not remove {file_path}: {e}")
    
    if cleaned_count > 0:
        print(f"🧹 Cleaned up {cleaned_count} temporary image files")


def display_final_summary(results, image_path, output_dir):
    """Display clean final pipeline summary"""
    seraphine_analysis = results['seraphine_analysis']
    
    print(f"\n📊 PIPELINE SUMMARY:")
    print("=" * 50)
    print(f"  📸 Image: {os.path.basename(image_path)}")
    print(f"  🎯 YOLO detections: {len(results['yolo_detections'])}")
    print(f"  📝 OCR detections: {len(results['ocr_detections'])}")
    print(f"  🔗 Merged detections: {len(results['merged_detections'])}")
    print(f"  🧠 Seraphine groups: {seraphine_analysis['analysis']['total_groups']}")
    print(f"  📊 Grouping efficiency: {seraphine_analysis['analysis']['grouping_efficiency']:.1%}")
    
    if results.get('gemini_analysis'):
        print(f"  🤖 LLM-identified icons: {results['gemini_analysis']['total_icons_found']}")
        print(f"  ⏱️  Gemini analysis time: {results['gemini_analysis']['analysis_duration_seconds']:.2f}s")
    
    print(f"  ⏱️  Total processing time: {results['timing']['total_time']:.3f}s")
    
    # Show what was saved
    if config.get("save_json"):
        print(f"  💾 Final results saved to: {output_dir}/")
    if config.get("save_gemini_json") and results.get('gemini_analysis'):
        print(f"  📋 Gemini JSON saved separately")
    if config.get("save_gemini_visualization") and results.get('gemini_visualization_path'):
        print(f"  🎨 Gemini visualization created")
    if not config.get("save_images"):
        print(f"  🧹 Temporary images cleaned up")
    
    print(f"\n🚀 PIPELINE COMPLETE!")


def save_final_results_json(results: dict, output_dir: str, filename_base: str):
    """Save only the final comprehensive results JSON"""
    from datetime import datetime
    
    current_time = datetime.now().strftime("%H-%M")
    
    # Safely extract seraphine analysis data
    seraphine_analysis = results.get('seraphine_analysis', {})
    seraphine_analysis_data = seraphine_analysis.get('analysis', {})
    
    # Create comprehensive final results
    final_results = {
        'pipeline_info': {
            'filename': filename_base,
            'processing_timestamp': current_time,
            'total_processing_time_seconds': results['timing']['total_time'],
            'pipeline_version': '3.0_optimized'
        },
        'detection_summary': {
            'yolo_detections': len(results['yolo_detections']),
            'ocr_detections': len(results['ocr_detections']),
            'merged_detections': len(results['merged_detections'])
        },
        'seraphine_analysis': {
            'total_groups': seraphine_analysis_data.get('total_groups', 0),
            'horizontal_groups': seraphine_analysis_data.get('horizontal_groups', 0),
            'vertical_groups': seraphine_analysis_data.get('vertical_groups', 0),
            'grouping_efficiency': seraphine_analysis_data.get('grouping_efficiency', 0.0),
            'grouped_items': seraphine_analysis_data.get('grouped_items', 0),
            'ungrouped_items': seraphine_analysis_data.get('ungrouped_items', 0)
        },
        'gemini_analysis': results.get('gemini_analysis', {'analysis_completed': False}),
        'timing_breakdown': results['timing']
    }
    
    # Optionally include detailed groups if they exist
    if 'groups' in seraphine_analysis:
        final_results['seraphine_analysis']['detailed_groups'] = seraphine_analysis['groups']
    elif 'group_assignments' in seraphine_analysis:
        final_results['seraphine_analysis']['detailed_groups'] = seraphine_analysis['group_assignments']
    
    # Save final results
    final_path = os.path.join(output_dir, f"{filename_base}_final_results_{current_time}.json")
    with open(final_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"💾 Final results saved: {os.path.basename(final_path)}")
    return final_path


def save_gemini_results_json(gemini_analysis: dict, output_dir: str, filename_base: str):
    """Save Gemini analysis results as separate JSON file"""
    if not gemini_analysis:
        return
    
    from datetime import datetime
    current_time = datetime.now().strftime("%H-%M")
    
    gemini_path = os.path.join(output_dir, f"{filename_base}_gemini_analysis_{current_time}.json")
    with open(gemini_path, 'w', encoding='utf-8') as f:
        json.dump(gemini_analysis, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Gemini analysis saved: {os.path.basename(gemini_path)}")
    return gemini_path


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())