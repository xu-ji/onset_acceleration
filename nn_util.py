import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .util import *
from .nn_metrics import *

device = torch.device("cuda")


def get_nn_data(args, cond=False):
  # make 3 dataloaders, each serving up all data types - but not all are used
  with gzip.open(os.path.join(root_dir, "data/nn_data_full.gz"), "rb") as f:
    nn_data = pickle.load(f)

  # int_train, int_val
  n = nn_data["int_trainval_norm"]["ids"].shape[0]
  shuffle = np.random.permutation(n)
  for data_type in nn_data_types:
    nn_data["int_trainval_norm"][data_type] = nn_data["int_trainval_norm"][data_type][shuffle]

  n_split = int(np.ceil(n / args.crossval_folds))
  print("n {}, n_split {}".format(n, n_split))
  int_trainvals = {"train": [], "val": []}
  for fold in range(args.crossval_folds):
    val_start = fold * n_split
    if fold == args.crossval_folds - 1:
      val_end_excl = n
    else:
      val_end_excl = min((fold + 1) * n_split, n)
    print("fold {} start {} end excl {}".format(fold, val_start, val_end_excl))

    val_inds = np.arange(val_start, val_end_excl)
    train_inds = np.concatenate((np.arange(val_start), np.arange(start=val_end_excl, stop=n)), axis=0)
    ordered_tup_train = tuple(
      [torch.Tensor(nn_data["int_trainval_norm"][data_type][train_inds]) for data_type in nn_data_types])
    ordered_tup_val = tuple(
      [torch.Tensor(nn_data["int_trainval_norm"][data_type][val_inds]) for data_type in nn_data_types])

    int_trainvals["train"].append(torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*ordered_tup_train),
                                                              batch_size=args.batch_size, shuffle=True,
                                                              drop_last=False))
    int_trainvals["val"].append(torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*ordered_tup_val),
                                                            batch_size=args.batch_size, shuffle=True, drop_last=False))

  ordered_tup_int_test = tuple([torch.Tensor(nn_data["int_test_norm"][data_type]) for data_type in nn_data_types])
  int_test = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*ordered_tup_int_test),
                                         batch_size=args.batch_size, shuffle=True, drop_last=False)
  ordered_tup_ext_test = tuple([torch.Tensor(nn_data["ext_test_norm"][data_type]) for data_type in nn_data_types])
  ext_test = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*ordered_tup_ext_test),
                                         batch_size=args.batch_size, shuffle=True, drop_last=False)

  if not cond:
    return int_trainvals, int_test, ext_test
  else:
    return int_trainvals, int_test, ext_test, nn_data["int_trainval_norm"]["conditions"]


class NNModel(nn.Module):
  # https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py
  # each data type augmented with age, except demog

  def __init__(self, args, c_inds):
    super(NNModel, self).__init__()
    self.model_inputs = args.model_inputs
    self.feat_sz = args.feat_sz
    self.num_feat = len(self.model_inputs) * self.feat_sz
    self.c_inds = c_inds

    feat_modules = []
    for m in nn_module_order:
      if m in self.model_inputs:
        feat_modules.append(make_module(m, args))
      else:
        feat_modules.append(make_module("empty"))
    self.feat_modules = nn.ModuleList(feat_modules)

    self.head = make_module("head", args)

    self.n_params = sum(p.numel() for p in self.parameters())
    print("Created model with {} parameters".format(self.n_params))

  def forward(self, result, data):
    n = data[0].shape[0]
    feats = []
    for m in nn_module_order:
      if m in self.model_inputs:
        if m == "sbc":
          feat_input = (data[nn_data_types.index(m)], data[nn_data_types.index("age_imaging")])
        elif m == "bbc":
          feat_input = torch.cat([data[nn_data_types.index(m)], data[nn_data_types.index("age_imaging")]], dim=1)
        else:
          feat_input = data[nn_data_types.index(m)]
        feats.append(self.feat_modules[nn_module_order.index(m)](feat_input))

    feats = torch.cat(feats, dim=1)
    assert feats.shape == (n, self.num_feat)
    log_hazards = self.head(feats)
    assert log_hazards.shape == (n, num_conditions)

    # if self.training:
    #  print("log hazards")
    #  print(get_stats(log_hazards.cpu().numpy()))

    if result == "loss":
      survival_event = data[nn_data_types.index("observed")]
      survival_time = data[nn_data_types.index("diag_age")]
      return neg_cox_log_likelihood(log_hazards, survival_event, survival_time, self.c_inds)
    elif result == "log_hazards":
      return log_hazards
    else:
      raise NotImplementedError


def make_module(m, args=None):
  if m == "empty":
    return nn.Linear(1, 1)

  elif m in ["demog", "bb", "bbc", "blood"]:
    return nn.Sequential(
      nn.Linear(nn_module_input_len[m], 512),
      nn.ReLU(inplace=True),
      nn.Linear(512, 256),
      nn.ReLU(inplace=True),
      nn.Linear(256, args.feat_sz),
      nn.ReLU(inplace=True)  # relu terminated
    )

  elif m == "sbc":
    return SBCFeats(args)  # relu terminated

  elif m == "head":
    num_feat = len(args.model_inputs) * args.feat_sz  # 512 - 1280
    return nn.Sequential(
      nn.Linear(num_feat, 1024),
      nn.Dropout(args.dropout),
      nn.ReLU(inplace=True),
      nn.Linear(1024, 512),
      nn.Dropout(args.dropout),
      nn.ReLU(inplace=True),
      nn.Linear(512, num_conditions)  # hazards
    )

  else:
    raise NotImplementedError


class SBCFeats(nn.Module):
  def __init__(self, args):
    super(SBCFeats, self).__init__()

    print("NN version: {}".format(args.nn_version))

    if args.nn_version in [0, 1]:
      mult = 1
    elif args.nn_version == 2:
      mult = 0.5

    self.conv = nn.Sequential(
      nn.Conv1d(len(channels), int(mult * 64), kernel_size=5),
      nn.ReLU(inplace=True),
      nn.MaxPool1d(kernel_size=3, stride=2),

      nn.Conv1d(int(mult * 64), int(mult * 128), kernel_size=5),
      nn.ReLU(inplace=True),
      nn.MaxPool1d(kernel_size=3, stride=2),

      nn.Conv1d(int(mult * 128), int(mult * 64), kernel_size=3),
      nn.ReLU(inplace=True),
      nn.MaxPool1d(kernel_size=3, stride=2),

      nn.Conv1d(int(mult * 64), int(mult * 32), kernel_size=3),
      nn.ReLU(inplace=True),
      nn.MaxPool1d(kernel_size=3, stride=2),

      nn.Flatten(start_dim=1)
    )  # n, 32 * 23 approx - 32 * 19
    self.subfeat_len = int(mult * 608)

    if args.nn_version in [0, 2]:
      self.head = nn.Sequential(
        nn.Linear(self.subfeat_len + 1, 512),  # takes age
        nn.ReLU(inplace=True),
        nn.Linear(512, args.feat_sz),
        nn.ReLU(inplace=True)
      )
    elif args.nn_version == 1:
      self.head = nn.Sequential(
        nn.Linear(self.subfeat_len + 1, args.feat_sz),  # takes age
        nn.ReLU(inplace=True)
      )

  def forward(self, data):
    graph, age = data
    subfeat = self.conv(graph)
    # print("conv subfeat shape {}".format(subfeat.shape))
    assert subfeat.shape[0] == age.shape[0] and age.shape == (age.shape[0], 1)
    return self.head(torch.cat((subfeat, age), dim=1))


def train_nn(args, model, int_train, int_val):
  optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.gamma)

  best_model, best_epoch, best_metric = None, None, - np.inf
  train_results, val_results = {}, {}
  for epoch in range(args.max_epochs + 1):
    if epoch % args.lr_step == 0:
      print("Updated lr with scheduler")

    train_results[epoch] = eval_nn(args, model, int_train, epoch)
    val_results[epoch] = eval_nn(args, model, int_val, epoch)
    print("(train_nn eval_nn) {}: {} {}".format(epoch, train_results[epoch]["c_index"], datetime.now()))

    if len(val_results) == 1 or val_results[epoch]["c_index"] > best_metric:
      print("New best val!")
      best_metric = val_results[epoch]["c_index"]
      best_model = deepcopy(model).to("cpu")
      best_epoch = epoch

    if epoch == args.max_epochs:
      break

    model.train()
    for batch_idx, data in enumerate(int_train):
      data = tuple([d.to(device) for d in data])
      optimizer.zero_grad()
      loss = model("loss", data)
      loss.backward()

      if batch_idx % 100 == 0:
        print(
          "Epoch {} batch {}: loss {} lr {}, {}".format(epoch, batch_idx, loss.item(), optimizer.param_groups[-1]['lr'],
                                                        datetime.now()))
        grads = []
        for param in model.parameters():
          if param.grad is not None:
            grads.append(param.grad.view(-1))
        if len(grads) > 0:
          grads = torch.cat(grads).abs()
          print("Mean grad {} max {}".format(grads.mean().item(), grads.max().item()))
          sys.stdout.flush()
        else:
          print("Strangely no grad")

      assert torch.isfinite(loss)

      optimizer.step()
    scheduler.step()

  return best_model, best_epoch, train_results, val_results  # return all


def eval_nn(args, model, dl, epoch=-1, curr_device=device):
  # Loss (neg cox partial ll), C-index, Cox log rank p val, 40yr AUC, 50yr AUC, 60yr AUC
  # Averaged over 45 conditions

  data_keys = ["log_hazards", "survival_event", "survival_time"]  # , "loss", "c_index"
  batched_data = defaultdict(list)
  results = defaultdict(list)
  model.eval()
  for batch_idx, data in enumerate(dl):
    data = tuple([d.to(curr_device) for d in data])

    with torch.no_grad():  # non sigmoid pos_weight
      log_hazards_curr = model("log_hazards", data)  # diag is prob diag, age is continuous age

    survival_event_curr = data[nn_data_types.index("observed")]
    survival_time_curr = data[nn_data_types.index("diag_age")]

    for k in data_keys:
      kname = "{}_curr".format(k)
      batched_data[k].append(locals()[kname])

    if (len(batched_data[data_keys[0]]) == args.eval_n_batch) or (batch_idx == len(dl) - 1):
      # use batched data
      for k in data_keys:
        batched_data[k] = torch.cat(batched_data[k], dim=0)

      results["loss"].append(neg_cox_log_likelihood(batched_data["log_hazards"], batched_data["survival_event"],
                                                    batched_data["survival_time"], model.c_inds).cpu().numpy())
      hazards = torch.exp(batched_data["log_hazards"])
      c_index_curr = compute_c_index(hazards, batched_data["survival_event"], batched_data["survival_time"],
                                     compute_mean=False).cpu().numpy()
      assert c_index_curr.shape == (num_conditions,)
      if model.c_inds == []:
        results["c_index"].append(c_index_curr.mean())
      else:
        results["c_index"].append(c_index_curr[model.c_inds].mean())
      results["c_index_condition"].append(c_index_curr)

      hazards_np = hazards.cpu().numpy()
      survival_event_np = batched_data["survival_event"].cpu().numpy()
      survival_time_np = batched_data["survival_time"].cpu().numpy()
      AUC_curr = compute_AUC(50, hazards_np, survival_event_np, survival_time_np, compute_mean=False)
      assert AUC_curr.shape == (num_conditions,)
      # results["auc_50"].append(AUC_curr.mean())
      results["auc_50_condition"].append(AUC_curr)

      results["hazards"].append(hazards_np.flatten())
      results["log_hazards"].append(batched_data["log_hazards"].flatten().cpu().numpy())

      # restart
      batched_data = defaultdict(list)

  report = {}  # can't assign np to list defaultdict
  for k, v in results.items():
    if not ("condition" in k):
      if isinstance(v[0], np.ndarray) and len(v[0].shape) > 0:
        report[k] = np.concatenate(v, axis=0)
      else:
        report[k] = np.array(v)
      assert len(report[k].shape) == 1
    else:
      report[k] = np.stack(v, axis=0)
      assert len(report[k].shape) == 2 and report[k].shape[1] == num_conditions

  report["loss"] = report["loss"].mean()
  report["c_index"] = report["c_index"].mean()
  report["c_index_condition"] = report["c_index_condition"].mean(axis=0)

  auc_condition = np.zeros(num_conditions)
  for c in range(num_conditions):
    auc_c = report["auc_50_condition"][:, c]
    isfinite = np.isfinite(auc_c)
    if isfinite.sum() > 0:  # redundant but to make explicit
      auc_condition[c] = auc_c[isfinite].mean()
    else:
      auc_condition[c] = np.nan
  report["auc_50_condition"] = auc_condition  # overwrite
  report["auc_50"] = auc_condition[np.isfinite(auc_condition)].mean()  # new

  report["log_hazard_stats"] = get_stats(report["log_hazards"])
  report["hazard_stats"] = get_stats(report["hazards"])
  del report["log_hazards"]
  del report["hazards"]
  report["epoch"] = epoch

  return report


def summarize_results(cross_val_results, epochs):
  best_results = defaultdict(list)
  best_c_index = - np.inf
  best_fold = None
  for fold in range(len(cross_val_results)):
    best_val_epoch_fold = epochs[fold]
    result = cross_val_results[fold][1][best_val_epoch_fold]  # val results for the fold and its best epoch
    for k, v in result.items():
      best_results[k].append(v)  # collect best epoch results across folds

    if result["c_index"] > best_c_index:
      best_c_index = result["c_index"]
      best_fold = fold

  print("(summarize_results) best epoch for all val")
  avg_result = {}
  for k, v in best_results.items():
    print(k, v)
    if not ("stats" in k):
      arr = np.array(v)
      avg_result[k] = (arr.mean(axis=0), arr.std(axis=0))

  return avg_result, best_fold


def eval_nn_per_disease(args, model, dl, model_name, conditions):
  # Loss (neg cox partial ll), C-index, Cox log rank p val, 40yr AUC, 50yr AUC, 60yr AUC
  # Averaged over 45 conditions

  data_keys = ["log_hazards", "survival_event", "survival_time"]  # , "loss", "c_index"
  batched_data = defaultdict(list)
  results = defaultdict(list)
  model.eval()
  for batch_idx, data in enumerate(dl):
    data = tuple([d.to(device) for d in data])

    with torch.no_grad():  # non sigmoid pos_weight
      log_hazards_curr = model("log_hazards", data)  # diag is prob diag, age is continuous age

    survival_event_curr = data[nn_data_types.index("observed")]
    survival_time_curr = data[nn_data_types.index("diag_age")]

    for k in data_keys:
      kname = "{}_curr".format(k)
      batched_data[k].append(locals()[kname])

    if (len(batched_data[data_keys[0]]) == args.eval_n_batch) or (batch_idx == len(dl) - 1):
      # use batched data
      for k in data_keys:
        batched_data[k] = torch.cat(batched_data[k], dim=0)

      hazards = torch.exp(batched_data["log_hazards"])
      results["c_index"].append(compute_c_index(hazards, batched_data["survival_event"], batched_data["survival_time"],
                                                compute_mean=False).cpu().numpy())

      # restart
      batched_data = defaultdict(list)

  report = {}  # can't assign np to list defaultdict
  print("collate")
  for k, v in results.items():
    print(k, v[0].shape)
    report[k] = np.stack(v, axis=0)
    print(report[k].shape)
    assert len(report[k].shape) == 2 and report[k].shape[1] == num_conditions  # num batches, num disease

  res_report = []

  for c in range(num_conditions):
    report_c = {"model_name": model_name, "condition": conditions[c][0]}
    report_c["c_index"] = report["c_index"][:, c].mean()
    res_report.append(report_c)

  return res_report
