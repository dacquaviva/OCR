#!/bin/bash
set -e

function init_variables() {
    script_dir=$(dirname $(realpath "$0"))
    source $script_dir/../../../../../scripts/misc/checks_before_run.sh

    readonly POSTPROCESS_DIR="$TAPPAS_WORKSPACE/apps/h8/gstreamer/libs/post_processes/"
    readonly RESOURCES_DIR="$TAPPAS_WORKSPACE/apps/h8/gstreamer/general/ocr_light/resources"
    readonly CROPING_ALGORITHMS_DIR="$POSTPROCESS_DIR/cropping_algorithms"

    readonly DEFAULT_POSTPROCESS_SO="$POSTPROCESS_DIR/libyolo_hailortpp_post.so"
    readonly DEFAULT_NETWORK_NAME="detector_ocr_light"
    readonly DEFAULT_BATCH_SIZE="1"
    readonly DEFAULT_HEF_PATH="$RESOURCES_DIR/detector.hef"
    readonly DEFAULT_VIDEO_SOURCE="$RESOURCES_DIR/images/output_final.mp4"
    readonly DEFAULT_CROP_SO="$CROPING_ALGORITHMS_DIR/libdetection_croppers.so"

    # OCR
    readonly OCR_HEF_PATH="$RESOURCES_DIR/ocr.hef"
    readonly OCR_POST_SO="$POSTPROCESS_DIR/libocr_post.so"

    postprocess_so=$DEFAULT_POSTPROCESS_SO
    network_name=$DEFAULT_NETWORK_NAME
    input_source=$DEFAULT_VIDEO_SOURCE
    hef_path=$DEFAULT_HEF_PATH
    json_config_path="null"
    nms_score_threshold=0.3
    nms_iou_threshold=0.45
    crop_so=$DEFAULT_CROP_SO

    thresholds_str="nms-score-threshold=${nms_score_threshold} nms-iou-threshold=${nms_iou_threshold} output-format-type=HAILO_FORMAT_TYPE_FLOAT32"

    print_gst_launch_only=false
    additional_parameters=""
    stats_element=""
    debug_stats_export=""
    show_fps=false
}

function print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --show-fps    Show FPS information"
    echo "  --help        Display this help message"
}

function parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --show-fps)
                show_fps=true
                shift
                ;;
            --help)
                print_usage
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done
}

init_variables $@
parse_args $@

source_element="filesrc location=$input_source ! decodebin ! videoconvert"

OBJECT_DETECTION_PIPELINE="videoscale qos=false ! \
    queue leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! \
    hailonet hef-path=$hef_path scheduling-algorithm=1 $thresholds_str vdevice-key=1 ! \
    queue leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! \
    hailofilter so-path=$postprocess_so function-name=$network_name config-path=$json_config_path qos=false ! \
    queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0"

OCR_PIPELINE="queue leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! \
    hailonet hef-path=$OCR_HEF_PATH scheduling-algorithm=1 scheduler-threshold=1  vdevice-key=1 ! \
    queue leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! \
    hailofilter so-path=$OCR_POST_SO qos=false ! \
    queue leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0"

PIPELINE="${debug_stats_export} gst-launch-1.0 ${stats_element} \
    $source_element ! tee name=t hailomuxer name=hmux \
    t. ! queue leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! hmux. \
    t. ! $OBJECT_DETECTION_PIPELINE ! hmux. \
    hmux. ! queue leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! \
    hailocropper so-path=$crop_so function-name=all_detections internal-offset=false name=cropper \
    hailoaggregator name=agg \
    cropper. ! queue leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! agg. \
    cropper. ! $OCR_PIPELINE ! agg. \
    agg. ! queue leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! \
    fpsdisplaysink video-sink=fakesink name=hailo_display text-overlay=false sync=false"

if [ "$show_fps" = true ]; then
    PIPELINE+=" print-arm-load=true print-fps=true -v 2>&1 | grep -E 'hailo_display|ARM_LOAD'"
fi

echo "Running $network_name on video with object detection and OCR"
echo ${PIPELINE}

if [ "$print_gst_launch_only" = true ]; then
    exit 0
fi

eval ${PIPELINE}