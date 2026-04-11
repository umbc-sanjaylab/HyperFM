'''
Preprocess PACE L1B and L2 Cloud data with ground truth to extract image patches and save as .npy files.
'''
# import libraries
import numpy as np
import xarray as xr
import os
import csv

def _read_csv(infile):
    
    file = open(infile, "r")
    data = list(csv.reader(file, delimiter=","))
    file.close()
    return data[1:]

def read_pace_image_l1b(file_path):
    # Load required groups
    rad_ds = xr.open_dataset(file_path, group="observation_data")
    # Extract the image data
    rad_blue = rad_ds['rhot_blue'] # (119,H,W)
    rad_red  = rad_ds['rhot_red']  # (163,H,W)
    rad_swir = rad_ds['rhot_SWIR'] # (9,H,W)

    qual_blue = rad_ds['qual_blue']
    qual_red  = rad_ds['qual_red']
    qual_swir = rad_ds['qual_SWIR']

    # print(qual_blue.shape, qual_red.shape, qual_swir.shape)
    # print(rad_blue.shape, rad_red.shape, rad_swir.shape)

    img = np.concat((rad_blue,rad_red,rad_swir),axis =0) # 291,H,W
    # print("image shape: ", img.shape)
    qual = np.concat((qual_blue,qual_red,qual_swir),axis =0) # 291,H,W
    # print("qual shape: ", qual.shape)
    img_clean = np.where(qual==0,img,np.nan)

    # Convert to NumPy and squeeze (291, H, W)
    img_np = np.squeeze(img_clean).astype(np.float32)

    return img_np  # shape: (291, H, W)

def read_pace_image_l2_cld(file_path):
    # Load required groups
    gphy_ds = xr.open_dataset(file_path, group="geophysical_data")

    # Extract the image data
    cot = np.array(gphy_ds['cot_21']) # (H, W)
    # print("COT: ", cot.shape)

    cer = np.array(gphy_ds['cer_21']) # (H, W)
    # print("CER: ", cer.shape)

    cwp = np.array(gphy_ds['cwp_21']) # (H, W)
    # print("CWP: ", cwp.shape)

    cth = np.array(gphy_ds['cth']) # (H, W)
    # print("CTH: ", cth.shape)    

    # return {
    #     'cot':np.squeeze(cot).astype(np.float32),
    #     'cer':np.squeeze(cer).astype(np.float32),
    #     'cwp':np.squeeze(cwp).astype(np.float32),
    #     'cth':np.squeeze(cth).astype(np.float32)
    # }
    img = np.stack([cot,cer,cwp,cth],axis =0)
    return np.squeeze(img).astype(np.float32) # shape: (4, H, W)

def read_pace_image_l2_cmask(file_path):
    # Load required groups
    gphy_ds = xr.open_dataset(file_path, group="geophysical_data")
    # Extract the image data
    cloud_flag_data = gphy_ds['cloud_flag'].values
    cloud_flag_dilated_data = gphy_ds['cloud_flag_dilated'].values
    img = np.stack([cloud_flag_data,cloud_flag_dilated_data],axis =0)
    return np.squeeze(img).astype(np.float32) # shape: (2, H, W)

def extract_img_from_granule(N_granule,img_size,stride):
    #granule shape (c, h,w)
    granule = np.transpose(N_granule, (1, 2, 0))

    h, w, c = granule.shape
    images = []

    for i in range(0, h - img_size + 1, stride):
        for j in range(0, w - img_size + 1, stride):
            patch = granule[i:i + img_size, j:j + img_size]
            images.append(patch)
    return images

def nan_percentage_exceeds_threshold(array, threshold=0.05):
    total_values = array.size
    nan_values = np.isnan(array).sum()
    return (nan_values / total_values) > threshold

def data_preprocess(granule_dir1,granule_dir2, csv_file, out_dir1,out_dir2,output_csv_path,threshold=0.01):
    # gran_list = [        
    #     fname for fname in os.listdir(granule_dir)
    #     if fname.lower().endswith(('.nc'))
    # ]
    output_rows = []
    gran_list = _read_csv(csv_file)
    # gran_list = ['PACE_OCI.20240710T112454.L1C.V3.5km.nc']
    for i in range(len(gran_list)):
        fname        = gran_list[i][0]
        filepath     = os.path.join(granule_dir1, fname)
        gran_np      = read_pace_image_l1b(filepath)  #(c,h,w)
        image_list   = extract_img_from_granule(gran_np,96,96)  # list of patch of shape(291,96,96)

        # print(gran_np.shape)

        fname1        = gran_list[i][1]
        filepath      = os.path.join(granule_dir2, fname1)
        gran_np_cld1  = read_pace_image_l2_cld(filepath)  #(c,h,w)
        cld_list      = extract_img_from_granule(gran_np_cld1 ,96,96)
        # print(gran_np_cld1.shape)

        fname2        = fname1.replace("L2.CLD.V3_1", "L2.CLDMASK.V3_1")
        filepath      = os.path.join(granule_dir2, fname2)
        gran_np_cld2  = read_pace_image_l2_cmask(filepath)  #(c,h,w)
        cldmask_list  = extract_img_from_granule(gran_np_cld2,96,96)
        # print(gran_np_cld2.shape)


        for r in range(len(image_list)):
            # check for nan percentage. allow 1% or less nan

            if not nan_percentage_exceeds_threshold(image_list[r], threshold):
                img_name1 = fname[:-3]+f"_image_{r:03d}.npy"
                img_name = os.path.join(out_dir1,img_name1)
                np.save(img_name, image_list[r])

                img_name2 = fname1[:-3]+f"_cld_{r:03d}.npy"
                img_name = os.path.join(out_dir2,img_name2)
                np.save(img_name, cld_list[r])

                img_name3 = fname2[:-3]+f"_cldmask_{r:03d}.npy"
                img_name = os.path.join(out_dir2,img_name3)
                np.save(img_name, cldmask_list[r])

                output_rows.append([img_name1,img_name2,img_name3])
        print("File Processed: ",i+1)


    # Write the output CSV
    header = ["rad","cld","cldmask"]
    with open(output_csv_path, 'w', newline='') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(header)
        writer.writerows(output_rows)

def verify(root_dir):
      
    image_list = [
        os.path.join(root_dir, fname)
        for fname in os.listdir(root_dir) if fname.lower().endswith('.npy')
    ]

    for f in image_list:
        img = np.load(f)  
        print(np.nanmax(img),np.nanmin(img)) 

if __name__ == '__main__':
    rad_dir   = '/umbc/rs/zzbatmos/users/ztushar1/pace_data_L1B' 
    cld_dir   = '/umbc/rs/zzbatmos/users/ztushar1/pace_data_l2' 
    csv_file  = 'data_split/matched_files.csv'
    out_dir1  = '/umbc/rs/zzbatmos/users/ztushar1/preprocessed_npy_rad_oneR2' 
    out_dir2  = '/umbc/rs/zzbatmos/users/ztushar1/preprocessed_npy_cld_oneR2'
    output_csv_path = "data_split/oneR2_npy_list.csv"

    os.makedirs(out_dir1, exist_ok=True)
    os.makedirs(out_dir2, exist_ok=True)

    data_preprocess(rad_dir,cld_dir,csv_file,out_dir1,out_dir2,output_csv_path)

    # verify(out_dir1)
    # verify(out_dir2)
    # image_path   = 'PACE_OCI.20240510T002414.L1B.V3.nc'
    # img = read_pace_image_l1b(os.path.join(root_dir,image_path))

    # 
    # image_path   = 'PACE_OCI.20240510T002418.L2.CLD.V3_0.nc'
    # img = read_pace_image_l2_cld(os.path.join(root_dir,image_path))
    # print(img.shape)