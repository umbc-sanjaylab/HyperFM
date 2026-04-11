# requirements
# pip install earthaccess xarray netCDF4 matplotlib numpy

from earthaccess import login, DataGranules, Store
import xarray as xr
import os
from datetime import datetime

def download_pace_oci_l1b(date_start: str, date_end: str = None, output_dir: str = "pace_data", max_granules: int = None):
    """
    Downloads PACE-OCI Level-1B data for a given date or range and loads all radiance bands.

    Parameters:
    - date_start (str): Start date (e.g., '2025-06-10' or '2025-06-10T12:00:00')
    - date_end (str): Optional end date for range (same format as start)
    - output_dir (str): Directory to save downloaded files

    Returns:
    - list of xarray.Dataset: List of full L1B datasets from downloaded files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Earthdata login
    auth = login(persist=True)

    # Step 2: Search granules
    if not date_end:
        date_end = date_start
    granules = DataGranules().short_name("PACE_OCI_L1B_SCI") \
        .temporal(date_start, date_end) \
        .cloud_hosted(True) \
        .get()

    print(f"Found {len(granules)} granules for {date_start} to {date_end}")
    if max_granules:
        granules = granules[:max_granules]
    print(f"Downloading {len(granules)} granules from {date_start} to {date_end}")

    if not granules:
        print("No granules found for the specified date range.")
        return []

    # Step 3: Download all granules
    local_files = Store(auth).get(granules, local_path=output_dir)


    # Step 4: Load each file into xarray
    datasets = []
    for file in local_files:
        print(f"Loading {file}")
        ds = xr.open_dataset(file)
        datasets.append(ds)

    return datasets

def download_pace_l2_product(
    date_start: str,
    date_end: str = None,
    output_dir: str = "pace_data",
    product: str = "CLD",  # options: "CLD" or "AER_UAA"
    max_granules: int = None
):
    """
    Downloads PACE Level-2 cloud or aerosol data for a given date or range.

    Parameters:
    - date_start (str): Start date (e.g., '2025-06-10')
    - date_end (str): Optional end date for range (same format as start)
    - output_dir (str): Directory to save downloaded files
    - product (str): "CLD" for clouds, "AER_UAA" for aerosols
    - max_granules (int): Optional limit on number of granules to download

    Returns:
    - list of xarray.Dataset: List of loaded datasets
    """
    os.makedirs(output_dir, exist_ok=True)

    auth = login(persist=True)

    if not date_end:
        date_end = date_start

    # Map product name to short_name
    product_map = {
        "CLD": "PACE_OCI_L2_CLOUD",
        "AER_UAA": "PACE_OCI_L2_AER_UAA",
        "CMK": "PACE_OCI_L2_CLOUD_MASK"
    }

    if product not in product_map:
        raise ValueError(f"Unsupported product: {product}")

    short_name = product_map[product]

    granules = DataGranules().short_name(short_name) \
        .temporal(date_start, date_end) \
        .cloud_hosted(True) \
        .get()

    print(f"Found {len(granules)} granules for product '{product}' from {date_start} to {date_end}")
    if max_granules:
        granules = granules[:max_granules]
    print(f"Downloading {len(granules)} granules")

    if not granules:
        print("No granules found.")
        return []

    local_files = Store(auth).get(granules, local_path=output_dir)

    datasets = []
    for file in local_files:
        print(f"Loading {file}")
        try:
            ds = xr.open_dataset(file)
            datasets.append(ds)
        except Exception as e:
            print(f"Error loading {file}: {e}")

    return datasets


if __name__ == '__main__':
    for d in ["2025-06-10"]:
        datasets = download_pace_oci_l1b(
            date_start=d,
            output_dir="hsi_data",
            max_granules=1
        )
        # Inspect first dataset if available
        if datasets:
            print(datasets[0])
        else:
            print("No Level 1b datasets were downloaded for the specified date range.")

        datasets = download_pace_l2_product(
            date_start=d,
            output_dir="target_data",
            product="CLD",  # CLD, CMK or "AER_UAA"
            max_granules=1
        )

        if datasets:
            print(datasets[0])
        else:
            print("No Level 2 datasets were downloaded.")

        datasets = download_pace_l2_product(
            date_start=d,
            output_dir="target_data",
            product="CMK",  # CLD, CMK or "AER_UAA"
            max_granules=1
        )

        if datasets:
            print(datasets[0])
        else:
            print("No Level 2 datasets were downloaded.")