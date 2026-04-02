import argparse


def str2bool(v):
    return v.lower() in ("true", "1")


arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


net_arg = add_argument_group("Network")
net_arg.add_argument("--net_depth", type=int, default=12)
net_arg.add_argument("--clusters", type=int, default=500)
net_arg.add_argument("--iter_num", type=int, default=0)
net_arg.add_argument("--net_channels", type=int, default=128)
net_arg.add_argument("--use_fundamental", type=str2bool, default=False)
net_arg.add_argument("--share", type=str2bool, default=False)
net_arg.add_argument("--use_ratio", type=int, default=0)
net_arg.add_argument("--use_mutual", type=int, default=0)
net_arg.add_argument("--ratio_test_th", type=float, default=0.9)
net_arg.add_argument("--sr", type=float, default=0.5)

data_arg = add_argument_group("Data")
data_arg.add_argument("--data_tr", type=str, default='/DATASET/SIFT/yfcc-sift-2000-train.hdf5')
data_arg.add_argument("--data_va", type=str, default='/DATASET/SIFT/yfcc-sift-2000-val.hdf5')
data_arg.add_argument("--data_te", type=str, default='/DATASET/SIFT/yfcc-sift-2000-test.hdf5')

obj_arg = add_argument_group("obj")
obj_arg.add_argument("--obj_num_kp", type=int, default=2000)
obj_arg.add_argument("--obj_top_k", type=int, default=-1)
obj_arg.add_argument("--obj_geod_type", type=str, default="episym", choices=["sampson", "episqr", "episym"])
obj_arg.add_argument("--obj_geod_th", type=float, default=1e-4)

loss_arg = add_argument_group("loss")
loss_arg.add_argument("--weight_decay", type=float, default=0)
loss_arg.add_argument("--momentum", type=float, default=0.9)
loss_arg.add_argument("--loss_classif", type=float, default=1.0)
loss_arg.add_argument("--loss_essential", type=float, default=0.5)
loss_arg.add_argument("--loss_essential_init_iter", type=int, default=20000)
loss_arg.add_argument("--geo_loss_margin", type=float, default=0.1)

train_arg = add_argument_group("Train")
train_arg.add_argument("--run_mode", type=str, default="train")
train_arg.add_argument("--train_lr", type=float, default=1e-3)
train_arg.add_argument("--scheduler", type=float, default=None)
train_arg.add_argument("--train_batch_size", type=int, default=32)
train_arg.add_argument("--gpu_id", type=str, default='0')
train_arg.add_argument("--num_processor", type=int, default=8)
train_arg.add_argument("--train_iter", type=int, default=500000)
train_arg.add_argument("--log_base", type=str, default="../log/")
train_arg.add_argument("--log_suffix", type=str, default="")
train_arg.add_argument("--val_intv", type=int, default=10000)
train_arg.add_argument("--save_intv", type=int, default=1000)

test_arg = add_argument_group("Test")
test_arg.add_argument("--use_ransac", type=str2bool, default=False)
test_arg.add_argument("--model_path", type=str, default="../log/train/")
test_arg.add_argument("--res_path", type=str, default="")

vis_arg = add_argument_group('Visualization')
vis_arg.add_argument("--tqdm_width", type=int, default=79)


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


def print_usage():
    parser.print_usage()