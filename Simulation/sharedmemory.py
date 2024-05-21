import subprocess
import time
import os
import multiprocessing.shared_memory as shared_memory
import numpy as np
import sys
import select
yaw = 0.0

def read_shared_memory():
    try:
        # Open the existing shared memory
        existing_shared_mem = shared_memory.SharedMemory(name='/aruco_shared_memory')
        
        while True:

            # Access the shared memory as a numpy array
            shared_array = np.ndarray((5,), dtype=np.float64, buffer=existing_shared_mem.buf)            
            if (shared_array[-1] > 0) or (shared_array[-1] < 0):
                # Print the values stored in shared memory
                print(f"Shared Memory Values - X: {shared_array[0]:.1f}, Y: {shared_array[1]:.1f}, Z: {shared_array[2]:.1f}, Yaw: {shared_array[3]:.1f}, Flag: {shared_array[-1]}")
                # Check for user input or termination signal
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0] or (shared_array[-1] == -1).any():
                    if shared_array[-1] == -1:
                        print("Sender stops sending: Quitting")
                    else:
                        print("User input 'q' or termination signal received: Quitting")
                    break
            elif(shared_array[-1]==0):
                print(f"Not target, Flag: {shared_array[-1]:.0f}")
            else:
                print("ERROR")
 
    except FileNotFoundError:
        print("Shared memory does not exist. Please ensure that it is created before running this script.")
    except Exception as e:
        print(f"Error while reading shared memory: {e}")
    existing_shared_mem.close()
    existing_shared_mem.unlink()
    

def start_aruco_pose():
    # Start aruco_pose.py as a subprocess with unbuffered output
    process = subprocess.Popen(
        ["python3", "-u", "aruco_pose.py"],  # -u flag for unbuffered stdout and stderr
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        #text=True,  # Ensure the output is returned as a string
        env=dict(os.environ, PYTHONUNBUFFERED="1")  # Unbuffered environment
    )
    return process


def main():
    print("Starting aruco_pose")
    process = start_aruco_pose()
    time.sleep(2)
    
    try:
        read_shared_memory()
    except KeyboardInterrupt:
        print("Stopping aruco_pose")
        time.sleep(1)
        shared_memory
        process.terminate()
        process.wait()
        print("aruco_pose terminated")
    except Exception as e:
        print(f'An error occurred: {e}')
        process.terminate()
        process.wait()
        print("aruco_pose terminated due to error")

if __name__ == "__main__":
    main()
