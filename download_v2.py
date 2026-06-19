import gdown
import os
import json
import concurrent.futures

base = r"D:\claude code\document\data\arran"

# Load file list from earlier JSON
with open(os.path.join(base, "filelist.json"), "r") as f:
    raw = f.read()
    start = raw.index("[")
    files = json.loads(raw[start:])

# Group by folder
from collections import defaultdict
by_folder = defaultdict(list)
for item in files:
    folder = item["path"].split("/")[0]
    by_folder[folder].append(item)

# First pass: download only CSV files (tiny, essential)
csv_files = [item for item in files if item["path"].endswith(".csv")]
print(f"CSV files to download: {len(csv_files)}")
for item in csv_files:
    folder, rest = item["path"].split("/", 1)
    dest_dir = os.path.join(base, folder)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, rest)
    if os.path.exists(dest):
        print(f"  SKIP: {item['path']}")
        continue
    print(f"  DOWNLOAD: {item['path']}")
    try:
        gdown.download(item["url"], dest, quiet=True)
        print(f"    OK ({os.path.getsize(dest)} bytes)")
    except Exception as e:
        print(f"    FAILED: {e}")

print("CSV download complete.")

# Second pass: download images (up to 4 parallel per folder)
print("\nDownloading images (4 parallel)...")
for folder_name, folder_items in sorted(by_folder.items()):
    image_items = [item for item in folder_items if not item["path"].endswith(".csv")]
    print(f"\n{folder_name}: {len(image_items)} image files")
    dest_dir = os.path.join(base, folder_name)
    os.makedirs(dest_dir, exist_ok=True)
    
    downloaded = 0
    failed = 0
    
    def download_item(item):
        nonlocal downloaded, failed
        url = item["url"]
        # Remove folder name prefix from path to get relative path inside folder
        rel_path = "/".join(item["path"].split("/")[1:])  # skip folder name
        dest = os.path.join(dest_dir, rel_path)
        if os.path.exists(dest):
            downloaded += 1
            return f"SKIP: {item['path']}"
        try:
            gdown.download(url, dest, quiet=True)
            downloaded += 1
            return f"OK: {item['path']}"
        except Exception as e:
            failed += 1
            return f"FAIL: {item['path']}: {e}"
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(download_item, item) for item in image_items]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if downloaded % 20 == 0:
                print(f"  [{downloaded}/{len(image_items)}] {result}")
    
    print(f"  {folder_name}: {downloaded} downloaded, {failed} failed")

print("\nAll done!")
