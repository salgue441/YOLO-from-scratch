from __future__ import division

import torch
import torch.nn as nn
from utils import parse_cfg, create_modules


class Darkent(nn.Module):
    def __init__(self, cfgfile):
        super(Darkent, self).__init__()

        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA: bool = True):
        modules = self.blocks[1:]
        outputs = {}
        write = 0

        for idx, module in enumerate(modules):
            module_type = module["type"]

            if module_type == "convolutional" or module_type == "upsample":
                x = self.modules_list[idx](x)

            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - idx

                if len(layers) == 1:
                    x = outputs[idx + layers[0]]

                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - idx

                    map1 = outputs[idx + layers[0]]
                    map2 = outputs[idx + layers[1]]

                    x = torch.cat((map1, map2), 1)

            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[idx - 1] + outputs[idx + from_]
