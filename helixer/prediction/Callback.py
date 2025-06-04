from lightning.fabric import Fabric


class Callback:
    """Base callback class. Only functions used during training accept/need the runner to share parameters
       while keeping flexibility. The runner is passed in by using for example
       fabric.call("on_train_epoch_end", runner=self, model=...) in the Runner class itself."""
    def on_fit_start(self, fabric: Fabric) -> None:
        """called at the very start of fitting/the entire training session"""

    def on_train_epoch_start(self, fabric: Fabric) -> None:  # undecided
        """called at the start of an epoch"""

    def on_train_batch_end(self, runner) -> None:
        """called at the end of each training batch pass through the model"""

    def on_train_epoch_end(self, runner) -> None:
        """called at the end of a 'train' epoch, meaning after all training batches are run through,
           but before running the final validation epoch"""

    def on_validation_epoch_start(self, fabric: Fabric) -> None:  # undecided
        """called at the start of a validation epoch, they either happen multiple times during a
           training epoch and at the end of the training epoch or just at the end"""

    def on_validation_batch_end(self, runner) -> None:
        """called at the end of each validation batch pass through the model"""

    def on_validation_epoch_end(self, runner) -> None:
        """called at the end of a 'validation' epoch, meaning after all validation batches are run
           through the model which depending on the training loop can happen multiple times during
           a training epoch"""

    def on_train_end(self, fabric: Fabric, model) -> None:  # undecided
        """called at the end of training, after the training epoch and the final validation epoch
           are finished"""

    def on_test_batch_end(self, runner) -> None:
        """called at the end of each test batch pass through the model"""

    def on_test_epoch_end(self, runner) -> None:
        """called at the end of the entire test epoch"""
