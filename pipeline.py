import subprocess
import sys
import os

def run_script(script_name):
    print(f"\n>>> Running {script_name}...")
    result = subprocess.run([sys.executable, script_name], capture_output=False)
    if result.returncode != 0:
        print(f"❌ Error in {script_name}. Aborting.")
        sys.exit(1)

def main():
    print("🚀 Starting MediMind AI Ingestion Pipeline")
    print("==========================================")
    
    # 1. Load Data
    run_script("step1_load_data.py")
    
    # 2. Chunking
    run_script("step2_chunking.py")
    
    # 3. Embeddings
    run_script("step3_embeddings.py")
    
    print("\n✅ PIPELINE COMPLETE!")
    print("You can now run: streamlit run step5_app.py")

if __name__ == "__main__":
    main()
