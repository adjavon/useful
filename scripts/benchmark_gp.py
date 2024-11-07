import gunpowder as gp
from funlib.persistence import Array
from torch.utils.data import DataLoader
from tqdm import tqdm
from useful.gp import GPDataset
import numpy as np


def run_benchmark():
    pipeline = gp.ArraySource(
        key=gp.ArrayKey("RAW"),
        array=Array(data=np.zeros((100, 100, 100), dtype=np.uint8)),
    )

    gp_request = gp.BatchRequest(
        {
            gp.ArrayKey("RAW"): gp.ArraySpec(roi=gp.Roi((0, 0, 0), (10, 10, 10))),
        }
    )

    dataset = GPDataset(pipeline, request=gp_request, verbose=False)
    dataloader = DataLoader(dataset, batch_size=16, num_workers=4)
    for i, sample in tqdm(enumerate(dataloader), total=10):
        assert isinstance(sample, dict)
        if i > 10:
            break


if __name__ == "__main__":
    print("Running benchmark for gunpowder")
    run_benchmark()
