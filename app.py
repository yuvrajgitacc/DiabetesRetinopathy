"""
OptiGemma — Main Flask Application
The orchestrator that ties all engines together.
"""
import os
import uuid
import json
import time
import cv2
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from config import FLASK_SECRET, DEBUG, UPLOAD_DIR, RESULTS_DIR, ALLOWED_EXTENSIONS
from engine.preprocessor import preprocess_for_display
from engine.detector import predict
from engine.gradcam import generate_gradcam, get_heatmap_analysis
from engine.segmentor import segment_vessels
from engine.gemma_report import generate_report, translate_report

# ---------------------------------------------------------------------------
# Flask App Setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = FLASK_SECRET
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """Serve the main dashboard."""
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Main analysis endpoint.
    Receives an uploaded fundus image + patient info,
    runs the full OptiGemma pipeline, returns results.
    """
    start_time = time.time()

    # --- 1. Validate Upload ---
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    # Generate unique ID for this analysis
    analysis_id = str(uuid.uuid4())[:8]

    # Save uploaded file
    ext = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{analysis_id}_original.{ext}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    file.save(filepath)

    # Get patient info (optional)
    patient_info = {}
    if request.form.get("age"):
        patient_info["age"] = request.form.get("age")
    if request.form.get("diabetes_duration"):
        patient_info["diabetes_duration"] = request.form.get("diabetes_duration")
    if request.form.get("sugar_level"):
        patient_info["sugar_level"] = request.form.get("sugar_level")
    if request.form.get("hba1c"):
        patient_info["hba1c"] = request.form.get("hba1c")

    try:
        # --- 2. Preprocess Image ---
        processed = preprocess_for_display(filepath)
        model_input = processed["model_input"]           # Enhanced (for Grad-CAM/TF)
        model_input_raw = processed["model_input_raw"]   # Clean (for EfficientNet-B3)
        original = processed["original"]

        # Save original resized for display
        original_path = os.path.join(RESULTS_DIR, f"{analysis_id}_scan.png")
        cv2.imwrite(original_path, original)

        # --- 3. Run Detection (EfficientNet-B3 primary, TF fallback) ---
        detection_result = predict(model_input_raw)

        # --- 4. Generate Heatmap (Grad-CAM uses TF model) ---
        heatmap_path = os.path.join(RESULTS_DIR, f"{analysis_id}_heatmap.png")
        heatmap_overlay, heatmap_raw = generate_gradcam(model_input, original, save_path=heatmap_path)
        heatmap_analysis = get_heatmap_analysis(heatmap_raw)

        # --- 5. Vessel Segmentation (RishiSwethan) ---
        vessel_path = os.path.join(RESULTS_DIR, f"{analysis_id}_vessels.png")
        vessel_map, vessel_stats = segment_vessels(filepath, save_path=vessel_path)

        # --- 6. Generate Gemma-4 Report ---
        report, raw_response = generate_report(
            detection_result,
            heatmap_analysis,
            vessel_stats,
            patient_info if patient_info else None,
        )

        elapsed = round(time.time() - start_time, 2)

        # --- 7. Compile Response ---
        result = {
            "success": True,
            "analysis_id": analysis_id,
            "processing_time": elapsed,
            "detection": detection_result,
            "heatmap_analysis": heatmap_analysis,
            "vessel_stats": vessel_stats,
            "report": report,
            "images": {
                "original": f"/results/{analysis_id}_scan.png",
                "heatmap": f"/results/{analysis_id}_heatmap.png",
                "vessels": f"/results/{analysis_id}_vessels.png",
            },
        }

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/translate", methods=["POST"])
def translate():
    """Translate an existing report to another language."""
    data = request.get_json()
    report = data.get("report")
    language = data.get("language", "hindi")

    if not report:
        return jsonify({"error": "No report provided"}), 400

    translated = translate_report(report, language)
    return jsonify({"success": True, "report": translated, "language": language})


@app.route("/results/<filename>")
def serve_result(filename):
    """Serve result images."""
    return send_from_directory(RESULTS_DIR, filename)


@app.route("/uploads/<filename>")
def serve_upload(filename):
    """Serve uploaded images."""
    return send_from_directory(UPLOAD_DIR, filename)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  OptiGemma -- AI-Driven Predictive Retinal Suite")
    print("  http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=DEBUG, host="0.0.0.0", port=5000)
