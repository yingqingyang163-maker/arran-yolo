import subprocess, sys, os

# All 6 subfolder IDs from earlier enumeration
folders = {
    "DTM": "1gTBJj6bqaFhQzEK1DJ5jg4Y0W9OWgqQe",
    "Hillshade": "1VnWmZqepDBE95VfMoDe3KTljojyta3_G",
    "Local_dominance": "1N6nmtdLfd3K2yz0KBFJUw3pZYBmv4Dz8",
    "Open_Positive": "1Wm3bAogcKxeKu98OvZ0GMuYp5Uf12BuA",
    "Sky_View_Factor": "1XTfpqhAjRi23zaCzDQHxpnVsNjnO_pLM",
    "Slope": "1ED7dTFBN4iZTawfKKMrTj263qC2Dv-pD",
}

base = r"D:\claude code\document\data\arran"
os.makedirs(base, exist_ok=True)

for name, fid in folders.items():
    dest = os.path.join(base, name)
    os.makedirs(dest, exist_ok=True)
    print(f"Downloading {name} ({fid})...")
    result = subprocess.run([
        sys.executable, "-m", "gdown", "--folder", fid,
        "-O", dest
    ], capture_output=True, text=True, timeout=600, 
       env={**os.environ, "PATH": r"C:\Users\Lenovo\AppData\Roaming\Python\Python313\Scripts;" + os.environ.get("PATH", "")})
    if result.returncode == 0:
        print(f"  {name}: OK")
    else:
        print(f"  {name}: FAILED (exit {result.returncode})")
        print(f"  stderr: {result.stderr[-200:]}")
print("Done")
