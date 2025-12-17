
import os
import subprocess
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
IDA_PATH = r"D:\software\IDA\IDA_new\ida.exe"
BIN_DIR = os.path.abspath("bin")
RESULT_DIR = os.path.abspath("result1")  # Use result1 directory as specified
SCRIPT_PATH = os.path.abspath("super_enhanced_decompile.py")  # Use super enhanced script

def main():
    """
    Iterates through binary files in the 'bin' directory, and for each file,
    runs IDA Pro in batch mode to execute the decompile1.py script.
    The output for each binary is saved as a .jsonl file in the 'result' directory.
    """
    if not os.path.exists(IDA_PATH):
        logger.error(f"IDA Pro executable not found at: {IDA_PATH}")
        sys.exit(1)

    if not os.path.isdir(BIN_DIR):
        logger.error(f"Binary directory not found: {BIN_DIR}")
        sys.exit(1)

    if not os.path.isdir(RESULT_DIR):
        logger.info(f"Result directory not found. Creating it: {RESULT_DIR}")
        os.makedirs(RESULT_DIR)

    logger.info("Starting batch decompilation process...")

    # Filter for files that are likely binaries (no extension)
    binary_files = [f for f in os.listdir(BIN_DIR) if os.path.isfile(os.path.join(BIN_DIR, f)) and '.' not in f]
    
    if not binary_files:
        logger.warning("No binary files found in the bin directory.")
        return
    
    logger.info(f"Found {len(binary_files)} binary files to process.")
    
    success_count = 0
    error_count = 0
    skip_count = 0

    for i, filename in enumerate(binary_files, 1):
        binary_path = os.path.join(BIN_DIR, filename)
        output_jsonl_path = os.path.join(RESULT_DIR, f"{filename}.jsonl")

        # Skip if the result file already exists
        if os.path.exists(output_jsonl_path):
            logger.info(f"[{i}/{len(binary_files)}] Skipping '{filename}' - result file already exists.")
            skip_count += 1
            continue

        logger.info(f"[{i}/{len(binary_files)}] Processing file: {filename}")

        # Construct the IDA Pro command - Pass output path as script argument
        command = [
            IDA_PATH,
            "-A",  # Autonomous mode (no UI)
            "-S" + SCRIPT_PATH + f" {output_jsonl_path}",  # Script path with argument
            "-c",  # Close database when script finishes
            binary_path
        ]
        
        logger.debug(f"Command: {' '.join(command)}")
        
        try:
            # Execute the command with proper handling
            result = subprocess.run(
                command, 
                check=True, 
                capture_output=True, 
                text=True, 
                timeout=300  # 5 minute timeout per file
            )
            
            # Check if output file was actually created and has content
            if os.path.exists(output_jsonl_path) and os.path.getsize(output_jsonl_path) > 0:
                logger.info(f"[{i}/{len(binary_files)}] Successfully processed {filename}. Output saved to {output_jsonl_path}")
                success_count += 1
            else:
                logger.warning(f"[{i}/{len(binary_files)}] Processing completed but no output file generated for {filename}")
                if result.stdout:
                    logger.debug(f"IDA stdout: {result.stdout}")
                if result.stderr:
                    logger.debug(f"IDA stderr: {result.stderr}")
                error_count += 1
                
        except subprocess.TimeoutExpired:
            logger.error(f"[{i}/{len(binary_files)}] Timeout while processing {filename} (>5 minutes)")
            error_count += 1
        except subprocess.CalledProcessError as e:
            logger.error(f"[{i}/{len(binary_files)}] Failed to process {filename}.")
            logger.error(f"IDA Pro returned non-zero exit code: {e.returncode}")
            if e.stdout:
                logger.error(f"Stdout: {e.stdout}")
            if e.stderr:
                logger.error(f"Stderr: {e.stderr}")
            error_count += 1
        except Exception as e:
            logger.error(f"[{i}/{len(binary_files)}] An unexpected error occurred while processing {filename}: {e}")
            error_count += 1

    logger.info("Batch decompilation process finished.")
    logger.info(f"Summary: {success_count} successful, {error_count} errors, {skip_count} skipped, {len(binary_files)} total")
    
    if error_count > 0:
        logger.warning(f"There were {error_count} errors during processing. Check the logs above for details.")
    
    return success_count, error_count, skip_count

if __name__ == '__main__':
    main()
