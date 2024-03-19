import json
import glob
import os
from pathlib import Path
import sys
from typing import List
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, cpu_count

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import packed_dataset
from tokenizer import Tokenizer

# Filename for SlimPajama
slimpajama_sets = {
    "train": "train/chunk*/*",
    "validation": "validation/chunk*/*",
    "test": "test/chunk*/*",
}


def prepare_full(
    source_path: Path,
    tokenizer_path: Path,
    destination_path: Path,
    chunk_size: int,
    split: str="train",
    filenames_subset: List[str] = None,
    process_id: int = 0
) -> None:

    destination_path.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(tokenizer_path)

    # Use the provided filenames_subset or default to all filenames
    filenames = filenames_subset 
    
    if not filenames:
        raise RuntimeError(
            f"No files matching {slimpajama_sets[split]} found at {source_path}. \n"
            "Make sure you download the data..."
        )

    builder = packed_dataset.PackedDatasetBuilder(
        outdir=destination_path,
        prefix=f"{split}_slimpajama_{process_id}",  # Use process_id to differentiate builders
        chunk_size=chunk_size,
        sep_token=tokenizer.bos_id,
        dtype="auto",
        vocab_size=tokenizer.vocab_size,
    )

    for filepath in filenames:
        print(f"Processing {filepath}")
        import pyarrow.parquet as pq
        for rows in tqdm(pq.ParquetFile(filepath).iter_batches()):
            df = rows.to_pandas()
            for row in df.index:
                text = df["text"][row]
                set_name = df["meta"][row].get("redpajama_set_name")
                if set_name and set_name == "RedPajamaGithub":
                    continue # we don't want to include the github data
                text_ids = tokenizer.encode(text)
                builder.add_array(np.array(text_ids, dtype=builder.dtype))

    # we throw away the final corpus to avoid meaningless corpus filled with bos_ids, see https://github.com/jzhang38/TinyLlama/issues/83 for more details
    # builder.write_reminder()


def prepare(
    source_path: Path = Path("data/SlimmestPajama-627B"),
    tokenizer_path: Path = Path("data/llama"),
    destination_path: Path = Path("data/red_pajama_sample_parquet/"),
    chunk_size: int = 2049 * 1024,
    splits: list=["validation"],
    percentage: float = 1.0,
) -> None:
    import time

    for split in splits:
        filenames = glob.glob(os.path.join(source_path, slimpajama_sets[split]), recursive=True)
        filenames = filenames[:int(len(filenames) * percentage)]
        print(filenames)
        
        num_processes = cpu_count() 
        chunked_filenames = np.array_split(filenames, num_processes)

        processes = []
        start_time = time.time()

        source_path = Path(os.path.join(source_path.absolute(), split))
        destination_path = Path(os.path.join(destination_path.absolute(), split))
        print(source_path, destination_path)
        for i, subset in enumerate(chunked_filenames):
            if subset.size == 0:
                continue
            p = Process(target=prepare_full, args=(source_path, tokenizer_path, destination_path, chunk_size, split, list(subset), i))
            processes.append(p)
            p.start()

    for p in processes:
        p.join()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(prepare)