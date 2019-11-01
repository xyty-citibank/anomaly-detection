
from mmcv.runner.hooks import OptimizerHook



class OptimizerHook(OptimizerHook):
    def __init__(self):
        super(OptimizerHook, self).__init__()

    def after_train_iter(self, runner):
        losses = runner.outputs['loss']
        for loss in losses:
            runner.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip is not None:
                self.clip_grads(runner.model.parameters())
            runner.optimizer.step()



