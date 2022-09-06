from pytorch_lightning.callbacks import Callback


class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is started!")

    def on_train_end(self, trainer, pl_module):
        print("Training is done.")
