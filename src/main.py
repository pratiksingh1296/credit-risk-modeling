import subprocess
import os
import sys

def run_script(script_path):
    print(f"\n{'='*50}")
    print(f"RUNNING: {os.path.basename(script_path)}")
    print(f"{'='*50}")
    
    result = subprocess.run([sys.executable, script_path], capture_output=False)
    
    if result.returncode != 0:
        print(f"❌ Error occurred in {os.path.basename(script_path)}")
        return False
    return True

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the sequence of the pipeline
    pipeline_scripts = [
        "train.py",          # 1. Trains Baselines & Saves Splits
        "calibrate.py",      # 2. Fits Calibration (CV=5) on Baselines
        "evaluate.py",       # 3. Generates Comparison Tables & Plots
        "explainability.py", # 4. Generates SHAP Global/Local Interpretations
        "decisions.py"       # 5. Applies Risk Buckets & Policy Decisions
    ]

    for script in pipeline_scripts:
        full_path = os.path.join(current_dir, script)
        if not os.path.exists(full_path):
            print(f"❌ File not found: {full_path}")
            continue

        success = run_script(full_path)
        if not success:
            print("\n🛑 Pipeline halted due to error.")
            break
    else:
        print("\n✅ PIPELINE COMPLETE: All models trained, evaluated, and explained!")

if __name__ == "__main__":
    main()