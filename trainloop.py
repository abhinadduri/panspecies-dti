# import Loop from lightning
import pytorch_lightning as pl

# I am very confused how the Loops work. I think this runs the epochs as we need them for contrastive training,
# but I am not sure if "contrastive" will be passed to the training_step function in the model.
class ConPlexEpochLoop(pl.loops.TrainingEpochLoop):
    def __init__(self, min_steps = None, max_steps = -1, contrastive = False):
        self.super.__init__(min_steps, max_steps)
        self.contrastive = contrastive

    def run(self, *args, **kwargs):
        if self.skip:
            return self.on_skip()

        self.reset()
        self.on_run_start(*args, **kwargs)

        while not self.done:
            self.advance(*args, **kwargs)
        if self.contrastive == True: # if contrastive then iterate over data again with contrastive loss
            self.batch_progress.reset_on_run()
            while not self.done:
                # add contrastive to the kwargs
                kwargs['contrastive'] = True
                self.advance(*args, **kwargs)

        output = self.on_run_end()
        return output
