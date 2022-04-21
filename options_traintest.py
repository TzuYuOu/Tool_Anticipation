import argparse

str2bool = lambda arg: arg.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description="Train model for video-based surgical instrument anticipation.")
parser.register('type', 'bool', str2bool)

# input data
parser.add_argument('--data_folder', type=str, default='cholec80/resize-frames/')
parser.add_argument('--annotation_folder', type=str, default='cholec80/tool_annotations/')
parser.add_argument('--output_folder', type=str, default='output/experiments/')
parser.add_argument('--trial_name', type=str, default='anticipation')
# model
parser.add_argument('--num_class', type=int, default=3)
parser.add_argument('--num_ins', type=int, default=5)
parser.add_argument('--drop_prob', type=float, default=.2)
parser.add_argument('--horizon', type=int, default=2)
# training
parser.add_argument('--model', type=str, choices=['alexnet', 'resnet'], default='resnet')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--loss_scale', type=float, default=1e-2)
# logging
parser.add_argument('--model_save_freq', type=int, default=5)
parser.add_argument('--num_samples', type=int, default=1)
# testing
parser.add_argument('--test_folder', type=str, default='output/test/')
parser.add_argument('--model_folder', type=str, default='output/experiments/20220420-1728_horizon2_anticipation/models/')
parser.add_argument('--model_epoch', type=int, default=20)