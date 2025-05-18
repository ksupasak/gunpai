from concurrent.futures import ThreadPoolExecutor
import time

# Function to run in threads
def task(name):
    print(f"Task {name} started")
    time.sleep(1)
    print(f"Task {name} completed")
    return f"Result from task {name}"

# Use ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=3) as executor:
    # Submit tasks
    futures = [executor.submit(task, i) for i in range(100)]

    # Get results
    for future in futures:
        result = future.result()
        print(result)