from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import os

s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="us-east-1")

bucket = "aft-vbi-pds"
img_prefix = "bin-images/"
meta_prefix = "metadata/"

os.makedirs("bin-images", exist_ok=True)
os.makedirs("metadata", exist_ok=True)

# Collect list of all image keys first
paginator = s3.get_paginator("list_objects_v2")
keys = []

for page in paginator.paginate(Bucket=bucket, Prefix=img_prefix):
    for obj in page.get("Contents", []):
        filename = obj["Key"].split("/")[-1]
        if filename.endswith(".jpg"):
            keys.append(obj["Key"])
    if len(keys) >= 25000:
        break

# Define download function
def download_pair(key):
    filename = key.split("/")[-1]
    img_path = f"bin-images/{filename}"
    json_key = meta_prefix + filename.replace(".jpg", ".json")
    json_path = f"metadata/{filename.replace('.jpg', '.json')}"

    try:
        s3.download_file(bucket, key, img_path)
        s3.download_file(bucket, json_key, json_path)
        return filename
    except Exception as e:
        print(f"⚠️ Failed: {filename} ({e})")
        return None

# Parallel download (e.g. 10 at a time)
with ThreadPoolExecutor(max_workers=6) as executor:
    futures = [executor.submit(download_pair, k) for k in keys[:25000]]
    for f in as_completed(futures):
        result = f.result()
        if result:
            print(f"✅ Downloaded {result}")
