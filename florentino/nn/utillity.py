import sys


def print_batch(add_string: str,
                epoch, total_epoch, epoch_padding,
                batch, total_batches, batch_padding,
                bar_length=20):
    progress = (batch / total_batches)
    block = int(round(bar_length * progress))
    text = (f"\r{epoch: >{epoch_padding}}/{total_epoch} epoch: "
            f"[{'=' * block}>{' ' * (bar_length - block)}] {batch: >{batch_padding}}/{total_batches} | "
            f"{add_string}")
    sys.stdout.write(text)
    sys.stdout.flush()
