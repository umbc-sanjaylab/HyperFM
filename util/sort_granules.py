import os
import csv
import argparse
import re
from datetime import datetime
from collections import defaultdict

def extract_date_time(filename):
    match = re.search(r'PACE_OCI\.(\d{8})T(\d{6})', filename)
    if match:
        return match.group(1), match.group(2)
    return None, None

def list_files(directory, suffix):
    return sorted([f for f in os.listdir(directory) if f.endswith(suffix)])

def parse_time(hhmmss):
    return datetime.strptime(hhmmss, "%H%M%S")

def time_difference(t1, t2):
    return abs((parse_time(t1) - parse_time(t2)).total_seconds())

def match_files(l1b_files, l2cld_files, threshold):
    # Group L2 files by date
    l2_by_date = defaultdict(list)
    for f in l2cld_files:
        date, time = extract_date_time(f)
        if date and time:
            l2_by_date[date].append((f, time))

    matched_rows = []
    unmatched_l1b = []

    for l1b in l1b_files:
        date, l1b_time = extract_date_time(l1b)
        if not date or not l1b_time:
            matched_rows.append((l1b, ''))
            unmatched_l1b.append(l1b)
            continue

        candidates = l2_by_date.get(date, [])
        best_match = ''
        best_diff = float('inf')

        for l2cld, l2_time in candidates:
            diff = time_difference(l1b_time, l2_time)
            if diff <= threshold and diff < best_diff:
                best_match = l2cld
                best_diff = diff

        if best_match:
            matched_rows.append((l1b, best_match))
        else:
            matched_rows.append((l1b, ''))
            unmatched_l1b.append(l1b)

    return matched_rows, unmatched_l1b

def write_csv(filepath, rows, header):
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

def find_matching_granule(dir1, dir2, output_csv, unmatched_csv, threshold):
    l1b_files = list_files(dir1, 'L1B.V3.nc')
    l2cld_files = list_files(dir2, 'L2.CLD.V3_1.nc')

    matched_rows, unmatched_l1b = match_files(l1b_files, l2cld_files, threshold)

    write_csv(output_csv, matched_rows, ['L1B_File', 'L2_CLD_File'])
    write_csv(unmatched_csv, [(f,) for f in unmatched_l1b], ['Unmatched_L1B_File'])

    print(f"Total L1B files: {len(l1b_files)}")
    print(f"Matched: {len(l1b_files) - len(unmatched_l1b)}")
    print(f"Unmatched: {len(unmatched_l1b)}")
    print(f"Results written to {output_csv} and {unmatched_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Match L1B and L2CLD files by date and time')
    parser.add_argument('--dir1', help='Directory with L1B.V3.nc files')
    parser.add_argument('--dir2', help='Directory with L2.CLD.V3_1.nc files')
    parser.add_argument('--output', default='matched_files.csv', help='CSV output path for matched/all results')
    parser.add_argument('--unmatched', default='unmatched_l1b.csv', help='CSV for unmatched L1B files')
    parser.add_argument('--threshold', type=int, default=10, help='Time difference threshold in seconds')
    args = parser.parse_args()

    find_matching_granule(args.dir1, args.dir2, args.output, args.unmatched, args.threshold)
