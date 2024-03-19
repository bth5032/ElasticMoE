from contextlib import contextmanager
import torch

@contextmanager
def profile(*args, **kwargs):
    try:
        yield torch.cuda.nvtx.range_push(*args, **kwargs)
    finally:
        torch.cuda.nvtx.range_pop()

@contextmanager
def writing(*args, **kwargs):
    writer = None
    try:
        if args and isinstance(args[0], str):
            writer = torch.utils.tensorboard.SummaryWriter(*args, **kwargs)
            class PrintingWriter(torch.utils.tensorboard.SummaryWriter):
                def add_scalar(self, *args, **kwargs):
                    print(args, kwargs)
                    super().add_scalar(*args, **kwargs)
                def add_text(self, *args, **kwargs):
                    print(args, kwargs)
                    super().add_text(*args, **kwargs)
                def add_hparams(self, *args, **kwargs):
                    print(args, kwargs)
                    super().add_hparams(*args, **kwargs)
                def close(self):
                    print("Closing writer")
                    super().close()
                def flush(self):
                    print("Flushing writer")
                    super().flush()
                
            writer = PrintingWriter(*args, **kwargs)
            yield writer
        else:
            class NoOpWriter(torch.utils.tensorboard.SummaryWriter):
                def add_scalar(self, *args, **kwargs):
                    pass
                def add_text(self, *args, **kwargs):
                    pass
                def close(self):
                    pass
            writer = NoOpWriter()
            yield writer
    finally:
        writer.flush()
        writer.close()

