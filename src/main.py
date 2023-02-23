import torch
import copy

import utility
import data
import model
import loss
from option import args
from trainer import Trainer
from torchinfo import summary

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)


def main():
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            if args.model == 'UFSRCNN' and not args.test_only:
                downargs = copy.deepcopy(args)
                downargs.model = 'DFSRCNN'
                downargs.save = 'dfsrcnn_v1_x2'
                downargs.load = ''
                downargs.resume = 0
                downargs.reset = False
                downckp = utility.checkpoint(downargs)
                downmodel = model.Model(downargs, downckp)
            elif (args.model == 'UFSRCNNPS' or args.model == 'UFSRCNNPSV2' \
                    or args.model == 'UFSRCNNPSV6'  or args.model == 'UFSRCNNPSV7') \
                    and not args.test_only:
                downargs = copy.deepcopy(args)
                downargs.model = 'DFSRCNNPS'
                if args.scale[0] == 2:
                    downargs.save = 'dfsrcnnps_v1_x2'
                elif args.scale[0] == 3:
                    downargs.save = 'dfsrcnnps_v1_x3'
                elif args.scale[0] == 4:
                    downargs.save = 'dfsrcnnps_v1_x4'
                downargs.load = ''
                downargs.resume = 0
                downargs.reset = False
                downckp = utility.checkpoint(downargs)
                downmodel = model.Model(downargs, downckp)
            else:
                downmodel = None
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            # param = utility.count_param(_model, args.model)
            # print('%s total parameters: %.2fM (%d)' % (args.model, param / 1e6, param))
            _batch_size = args.batch_size if not args.test_only else 1
            checkpoint.write_log(str(
                summary(_model, 
                        input_size = [(_batch_size, args.n_colors, args.patch_size, args.patch_size), args.scale],
                        col_names = ("kernel_size", "output_size", "num_params", "params_percent", "mult_adds"),
                        depth = 3, 
                        mode = "eval", 
                        row_settings = ("depth", "var_names"))))
            # checkpoint.write_log(str(summary(_model, [(_batch_size, args.n_colors, 1280, 720), args.scale])))
            t = Trainer(args, loader, _model, _loss, checkpoint, downmodel)
            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()


if __name__ == '__main__':
    main()
