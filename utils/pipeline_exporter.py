"""
Pipeline Export Utility
Handles JSON export and data structure creation for the enhanced pipeline
"""
import os
import json
from datetime import datetime
from utils.helpers import debug_print

def create_enhanced_seraphine_structure(seraphine_analysis, original_merged_detections):
    """
    Create ENHANCED seraphine structure with Gemini results
    Format: H1_1: {bbox, g_icon_name, g_brief, m_id, y_id, o_id}
    """
    bbox_processor = seraphine_analysis.get('bbox_processor')
    if not bbox_processor:
        return {}
    
    # Create lookup from m_id to original detection data
    m_id_to_original = {}
    for detection in original_merged_detections:
        m_id = detection['m_id']
        m_id_to_original[m_id] = {
            'y_id': detection.get('y_id'),
            'o_id': detection.get('o_id'),
            'source': detection.get('source', 'unknown'),
            'type': detection.get('type', 'unknown')
        }
    
    enhanced_groups = {}
    
    # Process each group
    for group_id, boxes in bbox_processor.final_groups.items():
        enhanced_groups[group_id] = {}
        
        # Process each box in the group
        for i, bbox in enumerate(boxes):
            item_id = f"{group_id}_{i+1}"  # H1_1, H1_2, V2_1, etc.
            
            # Look up original IDs using m_id
            original_data = m_id_to_original.get(bbox.merged_id, {})
            
            enhanced_groups[group_id][item_id] = {
                'bbox': [bbox.x1, bbox.y1, bbox.x2, bbox.y2],
                'g_icon_name': getattr(bbox, 'g_icon_name', 'unanalyzed'),
                'g_brief': getattr(bbox, 'g_brief', 'Not analyzed'),
                'm_id': bbox.merged_id,  # M001, M002, etc.
                'y_id': original_data.get('y_id'),  # ✅ NOW PROPERLY LOOKED UP!
                'o_id': original_data.get('o_id'),  # ✅ NOW PROPERLY LOOKED UP!
                'type': original_data.get('type', bbox.bbox_type),  # Use original or fallback
                'source': original_data.get('source', bbox.source)   # Use original or fallback
            }
    
    return enhanced_groups

def save_enhanced_pipeline_json(image_path, detection_results, seraphine_analysis, gemini_results, config):
    """
    Save ENHANCED pipeline JSON with Gemini integration
    """
    if not config.get("save_json", False):
        debug_print("\n⏭️  JSON saving disabled in config")
        return None
    
    debug_print("\n💾 Saving Enhanced Pipeline JSON (with Gemini)")
    debug_print("=" * 70)
    
    # Create enhanced seraphine structure with PROPER ID TRACKING!
    enhanced_seraphine_groups = create_enhanced_seraphine_structure(
        seraphine_analysis, 
        detection_results['merged_detections']  # ✅ PASS ORIGINAL MERGED DETECTIONS!
    )
    
    # Choose the right field name based on Gemini success
    seraphine_field_name = "seraphine_gemini_groups" if gemini_results else "seraphine_groups"
    
    # Rest of save_pipeline_json logic but with enhanced structure
    output_dir = config.get("output_dir", "outputs")
    current_time = datetime.now().strftime("%d-%m")
    filename_base = os.path.splitext(os.path.basename(image_path))[0]
    
    analysis = seraphine_analysis['analysis']
    
    pipeline_results = {
        'pipeline_version': 'v1.2_enhanced_with_gemini',
        'timestamp': datetime.now().isoformat(),
        'image_info': {
            'filename': os.path.basename(image_path),
            'path': image_path
        },
        'detection_summary': {
            'yolo_count': len(detection_results['yolo_detections']),
            'ocr_count': len(detection_results['ocr_detections']),
            'merged_count': len(detection_results['merged_detections']),
            'total_input': len(detection_results['yolo_detections']) + len(detection_results['ocr_detections']),
            'merge_efficiency': f"{len(detection_results['yolo_detections']) + len(detection_results['ocr_detections']) - len(detection_results['merged_detections'])} removed"
        },
        'seraphine_summary': {
            'total_groups': analysis['total_groups'],
            'horizontal_groups': analysis['horizontal_groups'],
            'vertical_groups': analysis['vertical_groups'],
            'long_box_groups': analysis['long_box_groups'],
            'grouping_efficiency': analysis['grouping_efficiency']
        },
        'gemini_summary': {
            'enabled': bool(gemini_results),
            'total_icons_analyzed': gemini_results.get('total_icons_found', 0) if gemini_results else 0,
            'successful_analyses': gemini_results.get('successful_analyses', 0) if gemini_results else 0,
            'analysis_time': gemini_results.get('analysis_duration_seconds', 0) if gemini_results else 0
        },
        'timing_breakdown': {
            **detection_results['timing'],
            'seraphine_time': seraphine_analysis.get('seraphine_timing', 0),
            'gemini_time': gemini_results.get('analysis_duration_seconds', 0) if gemini_results else 0
        },
        'detections': {
            'yolo_detections': detection_results['yolo_detections'],
            'ocr_detections': detection_results['ocr_detections'], 
            'merged_detections': detection_results['merged_detections']
        },
        seraphine_field_name: enhanced_seraphine_groups  # DYNAMIC FIELD NAME!
    }
    
    # Add Gemini analysis metadata only if successful (avoid duplication)
    if gemini_results:
        pipeline_results['gemini_analysis_metadata'] = {
            'total_images_analyzed': gemini_results.get('total_images_analyzed', 0),
            'analysis_mode': gemini_results.get('analysis_mode', 'unknown'),
            'timestamp': gemini_results.get('analysis_timestamp')
        }
    
    # Save JSON file
    json_filename = f"{filename_base}_enhanced_v1_{current_time}.json"
    json_path = os.path.join(output_dir, json_filename)
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(pipeline_results, f, indent=2, ensure_ascii=False)
    
    debug_print(f"✅ Enhanced Pipeline JSON saved: {json_filename}")
    debug_print(f"   📊 Complete pipeline with Gemini integration")
    debug_print(f"   🔗 Perfect ID mapping: Y/O → M → Seraphine Groups → Gemini Analysis")
    debug_print(f"   🎯 Field name: '{seraphine_field_name}' (dynamic based on Gemini success)")
    
    return json_path
