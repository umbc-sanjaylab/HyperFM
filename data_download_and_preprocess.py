'''
# args: date list, max_granules per day, raw_data_dir, target_data_dir
# outputs: downloads data to specified dirs
'''
import argparse
def get_args_parser():
    parser = argparse.ArgumentParser('Pace Data Download and Preprocess', add_help=False)
    parser.add_argument('--date_list', default=["2025-06-10"], type=list,
                        help='List of dates to download data for')
    parser.add_argument('--raw_data_dir', default="raw_data", type=str)
    parser.add_argument('--preprocessed_data_dir', default="preprocessed_data", type=str)
    parser.add_argument('--max_granules', default=1, type=int,
                        help='Maximum number of granules to download per day')
    return parser

def main(args):
    from util.data_download import download_pace_oci_l1b, download_pace_l2_product
    import os

    raw_hsi_dir = os.path.join(args.raw_data_dir, "hsi")
    raw_target_dir = os.path.join(args.raw_data_dir, "target")
    match_csv_file = os.path.join(args.raw_data_dir, "matched_files.csv")
    unmatched_csv_file = os.path.join(args.raw_data_dir, "unmatched_files.csv")

    os.makedirs(raw_hsi_dir, exist_ok=True)
    os.makedirs(raw_target_dir, exist_ok=True)

    # download data for each date
    for d in args.date_list:
        download_pace_oci_l1b(
            date_start=d,
            output_dir=raw_hsi_dir,
            max_granules=args.max_granules
        )

        download_pace_l2_product(
            date_start=d,
            output_dir=raw_target_dir,
            product="CLD",  # CLD, CMK or "AER_UAA"
            max_granules=args.max_granules
        )

        download_pace_l2_product(
            date_start=d,
            output_dir=raw_target_dir,
            product="CMK",  # CLD, CMK or "AER_UAA"
            max_granules=args.max_granules
        )

    # match File timestamps and create csv
    from util.sort_granules import find_matching_granule
    find_matching_granule(raw_hsi_dir, raw_target_dir, 
    match_csv_file, unmatched_csv_file, 10)

    # preprocess data
    from util.data_preprocess_w_GT import data_preprocess

    preprocessed_hsi_dir = os.path.join(args.preprocessed_data_dir, "hsi")
    preprocessed_target_dir = os.path.join(args.preprocessed_data_dir, "target")
    output_csv_path = os.path.join(args.preprocessed_data_dir, "preprocessed_data_list.csv")

    os.makedirs(preprocessed_hsi_dir, exist_ok=True)
    os.makedirs(preprocessed_target_dir, exist_ok=True)
    data_preprocess(raw_hsi_dir, raw_target_dir, match_csv_file, preprocessed_hsi_dir, preprocessed_target_dir, output_csv_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pace Data Download and Preprocess', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

