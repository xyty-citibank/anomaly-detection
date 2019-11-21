
from mmcv.runner.hooks import OptimizerHook



class MyOptimizerHook(OptimizerHook):
    def __init__(self, grad_clip=None):
        super(MyOptimizerHook, self).__init__(grad_clip=None)

    def after_train_iter(self, runner):
        losses = runner.outputs['loss']
        i = 0
        for loss in losses:
            runner.optimizer[i].zero_grad()
            loss.backward()
            if self.grad_clip is not None:
                self.clip_grads(runner.model.parameters())
            runner.optimizer[i].step()
            i += 1



