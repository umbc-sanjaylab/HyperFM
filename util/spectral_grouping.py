

def make_band_groups():
    # band counts
    total_blue = 119
    total_red = 163
    total_swir = 9

    # index ranges (stacked order)
    blue = list(range(0, total_blue))                        # 0–118
    red = list(range(total_blue, total_blue + total_red))    # 119–281
    swir = list(range(total_blue + total_red, total_blue + total_red + total_swir))  # 282–290

    groups = []

    # first 8 groups
    for i in range(8):
        blue_chunk = blue[i*13:(i+1)*13]
        red_chunk = red[i*18:(i+1)*18]
        swir_chunk = swir[i*1:(i+1)*1]
        group = tuple(blue_chunk + red_chunk + swir_chunk)
        groups.append(group)

    # last group
    blue_last = blue[8*13:]
    red_last = red[8*18:]
    swir_last = swir[8*1:]
    last_group = tuple(blue_last + red_last + swir_last)
    groups.append(last_group)

    return groups

def make_linear_groups(total_channels=291):
    group_sizes = [32] * 8 + [35]  # First 8 groups of 32, last group of 35
    groups = []
    start = 0

    for size in group_sizes:
        group = tuple(range(start, start + size))
        groups.append(group)
        start += size

    return groups
    