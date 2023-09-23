import json
import os.path as osp
from enum import Enum

import numpy as np
from collections import defaultdict, OrderedDict
from tensorboardX import  SummaryWriter
# from torch.utils.tensorboard import SummaryWriter as SummaryWriter2

class ConfigEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {
                '$enum': o.__module__ + "." + o.__class__.__name__ + '.' + o.name
            }
        elif callable(o):
            return {
                '$function': o.__module__ + "." + o.__name__
            }
        return json.JSONEncoder.default(self, o)


class Logger(object):
    def __init__(self, args, log_dir, **kwargs):
        self.logger_path = osp.join(log_dir, 'scalars.json')
        if args.device == 'cpu':
            pass
            # self.tb_logger = SummaryWriter2(
            #     log_dir=osp.join(log_dir, 'tflogger'))
        else:
            self.tb_logger = SummaryWriter(
                logdir=osp.join(log_dir, 'tflogger'), **kwargs,)
        if args.test_only == 0: # not change by test
            self.log_config(vars(args))

        self.scalars = defaultdict(OrderedDict)

    def writer_log(self, msg, quiet_ter=False):
        if not quiet_ter:
            print(msg)
        log_filepath = osp.join(osp.dirname(self.logger_path), 'log_detail')
        with open(log_filepath, "a", encoding='utf-8') as fd:
            fd.write('%s\n' % msg)

    def add_scalar(self, key, value, counter):
        assert self.scalars[key].get(counter, None) is None, 'counter should be distinct'
        self.scalars[key][counter] = value
        self.tb_logger.add_scalar(key, value, counter)

    def log_config(self, variant_data):
        config_filepath = osp.join(osp.dirname(self.logger_path), 'configs.json')
        with open(config_filepath, "w", encoding='utf-8') as fd:
            # save_content + file_path + indent(2个缩进) + sort_by_initial +  ConfigEncoder_override
            json.dump(variant_data, fd, indent=2, sort_keys=True, cls=ConfigEncoder)

    def dump(self):
        with open(self.logger_path, 'a',encoding='utf-8') as fd:
            json.dump(self.scalars, fd, indent=2)