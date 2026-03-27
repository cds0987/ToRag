from tqdm import tqdm

def create_iterior(iterator):
    return tqdm(
        iterator,
        desc="🟢 Encoding",
        colour="green",
        ncols=120,
        dynamic_ncols=False,
        bar_format=(
            "{l_bar}{bar:40} | "
            "{n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}]"
        ),
    )