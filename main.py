from model.trainer.fsl_trainer import FSLTrainer
from model.utils import (
    set_gpu,
    get_command_line_parser,
    postprocess_args,
)

if __name__ == '__main__':
    parser = get_command_line_parser()
    args = parser.parse_args()
    set_gpu(args.gpu)
    args = postprocess_args(args)

    trainer = FSLTrainer(args)
    trainer.train()
    trainer.try_test()
    trainer.final_record()

    print(args.save_path)



