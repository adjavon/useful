import contextlib
import gunpowder as gp
import time
from torch.utils.data import IterableDataset


class GPDataset(IterableDataset):
    def __init__(self, pipeline, request, verbose=False):
        """
        Pytorch interface to a gunpowder pipeline and a synapse database.

        Parameters
        ----------
        pipeline gp.Pipeline
            A gunpowder pipeline from which to request data.
        request: gp.ArraySpec
            The specification for the request.
            This gives the base shape of what will be sampled, as well as the
            expected arrays returned (used for formatting).
        verbose: bool
            Whether to print out request times.
        """
        self.pipeline = pipeline
        self.request = request
        self.verbose = verbose

    def __iter__(self):
        """An infinite generator of crops."""
        with gp.build(self.pipeline):
            while True:
                with contextlib.ExitStack() as stack:
                    t0 = time.perf_counter()
                    batch = self.pipeline.request_batch(self.request)
                    t1 = time.perf_counter()
                    if self.verbose:
                        print(f"Request time: {t1 - t0:.2f}")
                    yield self.format_batch(batch)

    def format_batch(self, batch):
        output = {}
        for key, spec in self.request.array_specs.items():
            output[key.identifier] = batch[key].data.copy()
        return output
