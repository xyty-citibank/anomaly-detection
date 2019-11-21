
import datetime
from collections import OrderedDict

import torch
from mmcv.runner.hooks.logger.text import TextLoggerHook



class MyTextLoggerHook(TextLoggerHook):

    def _log_info(self, log_dict, runner):
        if runner.mode == 'train':
            log_str = 'Epoch [{}][{}/{}]\t, '.format(
                log_dict['epoch'], log_dict['iter'], len(runner.data_loader))
            for i, lr in enumerate(log_dict['lr']):
                log_str += 'lr_{}:{:.5f}\t'.format(i, lr)
            if 'time' in log_dict.keys():
                self.time_sec_tot += (log_dict['time'] * self.interval)
                time_sec_avg = self.time_sec_tot / (
                    runner.iter - self.start_iter + 1)
                eta_sec = time_sec_avg * (runner.max_iters - runner.iter - 1)
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                log_str += 'eta: {}, '.format(eta_str)
                log_str += ('time: {:.3f}, data_time: {:.3f}, '.format(
                    log_dict['time'], log_dict['data_time']))
                log_str += 'memory: {}, '.format(log_dict['memory'])
        else:
            log_str = 'Epoch({}) [{}][{}]\t'.format(log_dict['mode'],
                                                    log_dict['epoch'] - 1,
                                                    log_dict['iter'])
        log_items = []
        for name, val in log_dict.items():
            # TODO: resolve this hack
            # these items have been in log_str
            if name in [
                    'mode', 'Epoch', 'iter', 'lr', 'time', 'data_time',
                    'memory', 'epoch'
            ]:
                continue
            if isinstance(val, float):
                val = '{:.4f}'.format(val)
            log_items.append('{}: {}'.format(name, val))
        log_str += ', '.join(log_items)
        runner.logger.info(log_str)

    def log(self, runner):
        log_dict = OrderedDict()
        # training mode if the output contains the key "time"
        mode = 'train' if 'time' in runner.log_buffer.output else 'val'
        log_dict['mode'] = mode
        log_dict['epoch'] = runner.epoch + 1
        log_dict['iter'] = runner.inner_iter + 1
        # only record lr of the first param group
        log_dict['lr'] = [lr[0] for lr in runner.current_lr()]
        if mode == 'train':
            log_dict['time'] = runner.log_buffer.output['time']
            log_dict['data_time'] = runner.log_buffer.output['data_time']
            # statistic memory
            if torch.cuda.is_available():
                log_dict['memory'] = self._get_max_memory(runner)
        for name, val in runner.log_buffer.output.items():
            if name in ['time', 'data_time']:
                continue
            log_dict[name] = val

        self._log_info(log_dict, runner)
        self._dump_log(log_dict, runner)



