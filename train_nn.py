import numpy as np
import gzip
import argparse
import traceback

from .util import *
from .nn_util import *

parser = argparse.ArgumentParser(description='train_nn')
parser.add_argument("--model_inputs", type=str, nargs="+", default=["demog", "bb", "sbc", "bbc", "blood"])
parser.add_argument("--max_epochs", type=int, default=80)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--batch_size", type=int, default=256) # 128
parser.add_argument("--eval_n_batch", type=int, default=8) # 1280
parser.add_argument("--gamma", type=float, default=0.5) # lr decay
parser.add_argument("--lr_step", type=int, default=50) # epochs
parser.add_argument("--dropout", type=float, default=0.3) # dropout
parser.add_argument("--feat_sz", type=int, default=256) # per input type
parser.add_argument("--crossval_folds", type=int, default=5)
parser.add_argument("--nn_version", type=int, default=0)
parser.add_argument("--sing_cond", type=str, nargs="+", default=[])
args = parser.parse_args()
print(args)

set_seed(1)

fname = "{}/models/nn_{}_{}_{}_{}_{}_{}_{}.pgz".format(root_dir, "_".join(args.model_inputs), args.lr, args.weight_decay, args.dropout, args.feat_sz, args.nn_version, pretty(args.sing_cond))
print("fname {}".format(fname))
if os.path.exists(fname):
  print("Exists, exiting")
  exit(0)

# {train val dict}, int_test, ext_test
int_trainvals, int_test, ext_test, conditions = get_nn_data(args, cond=True)

c_inds = []
if args.sing_cond != []:
  for c1 in args.sing_cond:
    for ci, c2 in enumerate(conditions):
      if c1 == c2[0]:
        c_inds.append(ci)
  assert len(c_inds) == len(args.sing_cond)
print(conditions, c_inds)

cross_val_results = []
models = []
epochs = []
best_model, best_metric, best_epoch = None, -np.inf, None
try:
  for fold in range(args.crossval_folds):
    print("Fold {} {}".format(fold, datetime.now()))
    model = NNModel(args, c_inds)
    model = model.to(device)

    # results have "data_type" train/val etc and "epoch" in them if train/val
    model, epoch, train_results, val_results = train_nn(args, model, int_trainvals["train"][fold], int_trainvals["val"][fold])
    model = model.to(device)
    int_test_results = eval_nn(args, model, int_test)
    ext_test_results = eval_nn(args, model, ext_test)

    cross_val_results.append((train_results, val_results, int_test_results, ext_test_results))
    epochs.append(epoch)

    if val_results[epoch]["c_index"] > best_metric:
      best_metric = val_results[epoch]["c_index"]
      best_model = model.to("cpu")
      best_fold = fold
      best_epoch = epoch
except:
  print(traceback.format_exc())
  with gzip.open(fname, "wb") as f:
    pickle.dump({"args": args,
                 "cross_val_results": None, "epochs": None,
                 "avg_results": None, "best_fold": None,
                 "best_model": None, "best_epoch": None}, f) # empty
  print("Saving empty results")
  exit(0)

print("Finished folds")
avg_results, best_fold_2 = summarize_results(cross_val_results, epochs) # gets mean and stddev for each metric on val
assert best_fold == best_fold_2

print("Average val results over folds:")
print(avg_results)

with gzip.open(fname, "wb") as f:
  pickle.dump({"args": args,
                   "cross_val_results": cross_val_results, "epochs": epochs,
                   "avg_results": avg_results, "best_fold": best_fold,
                   "best_model": best_model, "best_epoch": best_epoch}, f)

print("Finished")