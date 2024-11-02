import geopandas as gpd
import contextily as ctx
import os
import os.path as osp
import pandas as pd
import time
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from joblib import Memory
warnings.filterwarnings('ignore', category=UserWarning, module='joblib')

# Initialize Memory with cache disabled
memory = Memory(None)  # Disable caching by passing None

data_root = r"Street View image folder path"  # Input folder
output_dir = r"Output OSM image folder path"  # Output folder
# If there are no errors, the following does not need to be changed
pixel_size = 0.2627
img_size = 256
final_size = 512
dis = pixel_size * img_size / 2

# Get all files in the input folder
files = [file for file in os.listdir(data_root) if file.endswith('.jpg') and 'depthmap' not in file]

# Create output folder
os.makedirs(output_dir, exist_ok=True)

# Extract latitude and longitude and create a GeoDataFrame
data = []
for file in files:
    file_name = file
# 40.75290377,-73.97958215_2022-08_mpyEVlsley4foc-t1ndVpA_d62_z3.jpg ,street view image name example.
    parts = file.split('_') 
    lat_lon = parts[0]
    lat, lon = map(float, lat_lon.split(','))

    # lat, lon = map(float, file.split('_')[:2])
    data.append([lon, lat, file_name])

df = pd.DataFrame(data, columns=['Lon', 'Lat', 'Name'])
gdf = gpd.GeoDataFrame(
    df.Name,
    geometry=gpd.points_from_xy(df.Lon, df.Lat)
).set_crs('EPSG:4326').to_crs(epsg=3857).values

def download_and_save(point_info):
    file_name, point = point_info
    out_file_path = osp.join(output_dir, file_name)
    if osp.exists(out_file_path):
        return

    try:
        center_x, center_y = point.x, point.y
        img, _ = ctx.bounds2img(
            center_x - dis, center_y - dis,
            center_x + dis, center_y + dis,
            source='https://tile.openstreetmap.org/{z}/{x}/{y}.png',
            ll=False,
            zoom=18,
            wait=2,
            max_retries=5,
        )

        h, w, _ = img.shape
        img = Image.fromarray(img, mode='RGBA').convert('RGB')
        img_crop = img.crop([
            (w - img_size) / 2, (h - img_size) / 2,
            (w + img_size) / 2, (h + img_size) / 2
        ])
        img_crop = img_crop.resize((final_size, final_size))
        img.save(out_file_path)
        time.sleep(1)
    except Exception as e:
        print(f"Error processing {file_name}: {e}")

# Use multithreading for downloading
with ThreadPoolExecutor(max_workers=20) as executor:
    futures = [executor.submit(download_and_save, point_info) for point_info in gdf]
    for future in tqdm(as_completed(futures), total=len(gdf)):
        future.result()
