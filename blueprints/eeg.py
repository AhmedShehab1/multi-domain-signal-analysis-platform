"""EEG classification blueprint"""

from flask import Blueprint, request, Response, jsonify
from uuid import uuid4
from threading import Thread
from queue import Queue
import os
import subprocess
import numpy as np
from werkzeug.exceptions import RequestEntityTooLarge
import time

# Create blueprint
eeg_bp = Blueprint('eeg', __name__)

# Global task storage
tasks = {}

def _load_results(path):
    """Load classification results from NPZ file"""
    data = np.load(path, allow_pickle=True)
    results = {}
    for key in data.files:
        value = data[key]
        if isinstance(value, np.ndarray):
            if value.shape == ():
                value = value.item()
            else:
                value = value.tolist()
        results[key] = value
    return results
def _summarize_results(result: dict) -> dict:
    """Convert raw pipeline output into compact summary stats."""
    summary: dict[str, float | int] = {}

    subject_pred = result.get("subject_prediction")
    if subject_pred is not None:
        summary["subject"] = int(subject_pred)

    subject_conf = result.get("subject_confidence")
    if subject_conf is not None:
        summary["confidence"] = float(subject_conf)

    sample_preds = result.get("sample_predictions")
    if sample_preds:
        preds = np.asarray(sample_preds)
        summary["segment_total"] = int(preds.size)
        summary["segments_healthy"] = int(np.sum(preds == 0))
        summary["segments_ad"] = int(np.sum(preds == 1))

    sample_probs = result.get("sample_probabilities")
    if sample_probs:
        probs = np.asarray(sample_probs, dtype=float)
        if probs.ndim == 1:
            probs = probs.reshape(-1, 1)
        if probs.shape[1] >= 1:
            summary["avg_healthy"] = float(np.nanmean(probs[:, 0]))
        if probs.shape[1] >= 2:
            summary["avg_ad"] = float(np.nanmean(probs[:, 1]))

    sample_conf = result.get("sample_confidences")
    if sample_conf:
        conf_arr = np.asarray(sample_conf, dtype=float)
        summary["avg_segment_confidence"] = float(np.nanmean(conf_arr))

    return summary

def _run(task_id, input_path, model):
    """Run the classification pipeline in a separate thread"""
    q = tasks[task_id]
    try:
        # Get the absolute path to the script
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                  "classify_individual_eeg.py")
        
        run_pipeline_cli = [
            "python", script_path,
            "--input", input_path,
            "--model", model,
            "--output", f"{input_path}_results.npz",
        ]
        proc = subprocess.Popen(run_pipeline_cli, stdout=subprocess.PIPE, 
                               stderr=subprocess.STDOUT, text=True)
        
        # Stream output
        for line in proc.stdout:
            q.put(line.rstrip())
        
        proc.wait()
        
        # If results file exists, load it
        results_path = f"{input_path}_results.npz"
        if os.path.exists(results_path):
            tasks[task_id] = {
                "result": _load_results(results_path),
                "status": "completed",
                "timestamp": time.time(),
            }
            # Clean up files
            try:
                os.remove(input_path)
                os.remove(results_path)
            except:
                pass
        else:
            tasks[task_id] = {"error": "Failed to generate results", "status": "failed"}
            
        q.put("DONE")
    except Exception as exc:
        q.put(f"ERROR: {exc}")
        tasks[task_id] = {"error": str(exc), "status": "failed"}
        q.put("DONE")

@eeg_bp.route('/classify', methods=['POST'])
def classify():
    """Upload and classify EEG data"""
    try:
        # Ensure upload directory exists
        tmp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        
        # Check if request has the file part
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        
        file = request.files['file']
        
        # If user does not select file, browser also
        # submits an empty part without filename
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Securely generate temporary filename
        tmp_path = os.path.join(tmp_dir, f"{uuid4().hex}_{file.filename}")
        
        # Save file
        file.save(tmp_path)
        
        # Get model parameter, default if not provided
        model = request.form.get("model", "P-11-F-5-Base")
        
        # Create task
        task_id = uuid4().hex
        tasks[task_id] = Queue()
        
        # Start processing in background
        Thread(target=_run, args=(task_id, tmp_path, model), daemon=True).start()
        
        return jsonify({"task_id": task_id})
    
    except RequestEntityTooLarge:
        return jsonify({"error": "File too large"}), 413
    
    except Exception as e:
        import traceback
        print(f"Error in /classify: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@eeg_bp.route('/stream/<task_id>')
def stream(task_id):
    """Stream classification progress"""
    print(f"Stream request received for task: {task_id}")
    
    if task_id not in tasks:
        print(f"Task {task_id} not found in tasks dictionary")
        return jsonify({"error": "unknown task"}), 404

    def event_stream():
        q = tasks[task_id]
        print(f"Queue object type: {type(q)}")
        
        if not isinstance(q, Queue):
            print(f"Task not a queue: {tasks[task_id]}")
            # Task is complete, return the status
            yield f"data: {tasks[task_id].get('status', 'unknown')}\n\n"
            yield f"data: DONE\n\n"
            return
            
        print("Starting event stream for task")
        while True:
            line = q.get()
            print(f"Sending line: {line}")
            yield f"data: {line}\n\n"
            if line == "DONE":
                break

    return Response(event_stream(), mimetype="text/event-stream")

@eeg_bp.route('/results/<task_id>')
def results(task_id):
    """Get classification results"""
    if task_id not in tasks:
        return jsonify({"error": "unknown task"}), 404

    task_data = tasks[task_id]
    
    if isinstance(task_data, Queue):
        return jsonify({"status": "running"}), 202
        
    if "error" in task_data:
        return jsonify({"status": "failed", "message": task_data["error"]}), 500

    if "result" in task_data:
        summary = _summarize_results(task_data["result"])
        return jsonify({"status": "completed", "data": summary})

    return jsonify({"status": "unknown"}), 500

# Clean up old tasks periodically
def setup_task_cleanup():
    """Set up periodic task cleanup to avoid memory leaks"""
    import threading
    import time
    
    def cleanup_old_tasks():
        while True:
            time.sleep(3600)  # Clean up every hour
            to_delete = []
            for task_id, task in tasks.items():
                if isinstance(task, dict) and task.get("status") in ("completed", "failed"):
                    # Mark tasks older than 1 hour for deletion
                    if "timestamp" in task and time.time() - task["timestamp"] > 3600:
                        to_delete.append(task_id)
                        
            for task_id in to_delete:
                del tasks[task_id]
                
    # Start the cleanup thread
    threading.Thread(target=cleanup_old_tasks, daemon=True).start()