from ..util import *
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import seaborn as sns

def render_condition(ci, condition_name, sex_c, results):
  sns.set_theme(style='white')  # white

  text_fields = list(results[sex_c[0]][0].keys())

  print("text subfields")
  print(render_text_subfields)

  text_row_count = 1
  for _, v in render_text_subfields.items():
    if isinstance(v, list):
      text_row_count += len(v)
    else:
      text_row_count += 1

  col_count = len(results) * len(results[sex_c[0]]) + 1
  print("text_row_count {} col_count {}".format(text_row_count, col_count))

  # latex table string
  fig, axs = plt.subplots(2, col_count, figsize=(16, 12), facecolor='w', height_ratios=[4, 1]) # 16 8, 16 12
  table = [[None] * col_count for _ in range(text_row_count)]
  table[0][0] = r"\underline{OnsetNet score quartile}"

  # find maximum slice width
  max_slice_width = -np.inf
  for s in results:
    for q, qdict in results[s].items():
      res_sbc = qdict["sbc"]
      res_max = (res_sbc[0] + res_sbc[1]).max() # one stddev rendered
      max_slice_width = max(max_slice_width, res_max)
  print("max_slice_width", max_slice_width)

  col_ind = 1
  for s in results:
    # find highest q to bold
    highest_qval = {}
    highest_q = {}
    for q, qdict in results[s].items():
      row_ind = 1
      for f in text_fields:
        res_f = qdict[f]

        if f == "sbc": continue

        if isinstance(render_text_subfields[f], list):
          for fsub_i, fsub in enumerate(render_text_subfields[f]):
            if not row_ind in highest_qval:
              highest_qval[row_ind] = - np.inf

            fsub_main = res_f[0][fsub_i]
            if not isinstance(fsub_main, str) and fsub_main > highest_qval[row_ind]:
              highest_qval[row_ind] = fsub_main
              highest_q[row_ind] = q

            row_ind += 1

        else:
          if not row_ind in highest_qval:
            highest_qval[row_ind] = - np.inf

          fsub_main = res_f[0]
          if not isinstance(fsub_main, str) and fsub_main > highest_qval[row_ind]:
            highest_qval[row_ind] = fsub_main
            highest_q[row_ind] = q

          row_ind += 1

    print(highest_qval)
    print(highest_q)

    for q, qdict in results[s].items():
      table[0][col_ind] = "Q{} {}".format(q + 1, {0: "Male", 1:" Female"}[s])
      table[0][col_ind] = r"\underline{" + table[0][col_ind]  + r"}"
      #print(s, q, col_ind, qdict)

      row_ind = 1
      for f in text_fields:
        res_f = qdict[f]
        if f == "sbc":
          # add graphs directly
          for ch_i, ch in enumerate(channels):
            ch_mean = res_f[0][ch_i, :]
            ch_std = res_f[1][ch_i, :]
            p = axs[1][col_ind].plot(ch_mean, np.arange(ch_mean.shape[0]), label=ch)
            axs[1][col_ind].fill_betweenx(range(ch_mean.shape[0]), ch_mean - ch_std,
                          ch_mean + ch_std, facecolor=p[-1].get_color(), alpha=0.15)

            if col_ind in [2, 3, 4, 6, 7, 8]: # 0, 1, 5 excluded
              prev_q_ch = results[s][q-1][f][0][ch_i, :]
              axs[1][col_ind].fill_betweenx(range(ch_mean.shape[0]), ch_mean, prev_q_ch, facecolor="k", alpha=0.3)

          axs[1][col_ind].set_xlim([0, max_slice_width])

        else:
          if isinstance(render_text_subfields[f], list):
            for fsub_i, fsub in enumerate(render_text_subfields[f]):
              table[row_ind][0] = fsub  # repeats
              fsub_main = res_f[0][fsub_i]
              fsub_sub = res_f[1][fsub_i]

              if isinstance(fsub_main, str):
                table[row_ind][col_ind] = "{} ({})".format(fsub_main, fsub_sub)
              else:
                #table[row_ind][col_ind] = "${:10.2f} \pm {:10.1f}$".format(fsub_main, fsub_sub)
                table[row_ind][col_ind] = "{} \pm {}".format(
                  np.format_float_scientific(fsub_main, precision=2, exp_digits=1),
                        np.format_float_scientific(fsub_sub, precision=1, exp_digits=1))

                if highest_q[row_ind] == q:
                  print("setting bold", fsub)
                  table[row_ind][col_ind] = r"\mathbf{" + table[row_ind][col_ind] + "}"

                table[row_ind][col_ind] = r"$" + table[row_ind][col_ind] + "$"

              row_ind += 1

          else:
            fsub = render_text_subfields[f]
            table[row_ind][0] = fsub  # repeats
            fsub_main = res_f[0]
            fsub_sub = res_f[1]

            if isinstance(fsub_main, str):
              table[row_ind][col_ind] = "{} ({})".format(fsub_main, fsub_sub)

            elif f in ["hazard_ratio", "hazard_ratio_ful_adj"]:
              if q == 0:
                table[row_ind][col_ind] = "-"
              else:
                table[row_ind][col_ind] = "{} \,({})".format(
                  np.format_float_scientific(fsub_main, precision=2, exp_digits=1),
                  np.format_float_scientific(fsub_sub, precision=1, exp_digits=1))

                if highest_q[row_ind] == q:
                  print("setting bold", fsub)
                  table[row_ind][col_ind] = r"\mathbf{" + table[row_ind][col_ind] + "}"

                table[row_ind][col_ind] = r"$" + table[row_ind][col_ind] + "$"

            else:
              table[row_ind][col_ind] = "{} \pm {}".format(
                np.format_float_scientific(fsub_main, precision=2, exp_digits=1),
                  np.format_float_scientific(fsub_sub, precision=1, exp_digits=1))

              if highest_q[row_ind] == q:
                print("setting bold", fsub)
                table[row_ind][col_ind] = r"\mathbf{" + table[row_ind][col_ind] + "}"

              table[row_ind][col_ind] = r"$" + table[row_ind][col_ind] + "$"

            row_ind += 1

      col_ind += 1

  # add text table to figure
  for coli in range(col_count):
    for axi in range(2):
      if axi == 1:
        if coli == 0:
          #axs[axi][coli].set_axis_off()

          axs[axi][coli].spines[["left", "right", "top", "bottom"]].set_visible(False)
          axs[axi][coli].set_yticklabels([])
          axs[axi][coli].set_xticklabels([])
          curr_ylim = axs[axi][coli].get_ylim()  # start, end
          axs[axi][coli].text(0, curr_ylim[1], "SBC ($cm^3$)",
                              horizontalalignment='left',
                              verticalalignment='top')

          continue
        else:
          if coli == 1:
            axs[axi][coli].spines[["right", "top"]].set_visible(False)
            #axs[axi][coli].legend(loc='center left', bbox_to_anchor=(1, 0.5))
          else:
            axs[axi][coli].spines[["left", "right", "top"]].set_visible(False)
            axs[axi][coli].set_yticklabels([])
          #axs[axi][coli].get_legend().remove()

      else:
        ltx = table_latex(table, coli)
        #print(ltx)
        axs[axi][coli].set_axis_off()
        curr_ylim = axs[axi][coli].get_ylim()  # start, end
        axs[axi][coli].text(0, curr_ylim[1], ltx,
                            horizontalalignment='left',
                            verticalalignment='top'
                            )

  axs[1][0].legend(*axs[1][1].get_legend_handles_labels(), prop={'size':7}, loc="lower right") # loc=(0.8, 0.85)

  #for vline_col in [1, 5]:
  #  axs[0][vline_col].axvline(-0.5, color="dimgray", linestyle="-")  # end of this one #  linewidth=0.15

  #ylim_curr = axs[0][0].get_ylim()
  #axs[0][0].axhline(ylim_curr[1]*0.95, color="dimgray", linestyle="-")  # end of this one #  linewidth=0.15

  # add axis
  fig.suptitle(condition_name, y=1.02)
  plt.tight_layout()
  plt.savefig("{}/analysis/part_2_subtypes_sbc_{}.png".format(root_dir, ci), bbox_inches='tight')
  plt.savefig("{}/analysis/part_2_subtypes_sbc_{}.eps".format(root_dir, ci), bbox_inches='tight', format="eps")
  plt.close('all')


def table_latex(table, coli, vlines = [1, 5]):
  #col_count = len(table[0])
  if coli in vlines:
    header_string = r"\begin{tabular}{|c} "
  else:
    header_string = r"\begin{tabular}{c} "
  end_string = r" \end{tabular}"
  #print("header, end", header_string, end_string)

  res = header_string
  for ri, row in enumerate(table):
    row_s = row[coli] + r" \\"
    res += row_s
    #if ri == 16:
    #  break
  res += end_string
  return res


def compute_match_ind(linear_ids, ids): # ordered
  assert ids.shape == linear_ids.shape # 1-1 match
  id_match = np.expand_dims(linear_ids, axis=1) == np.expand_dims(ids, axis=0) # linear index, id index

  mx, my = np.nonzero(id_match) # my is index in ids per index in linear
  assert (mx == np.arange(linear_ids.shape[0])).all()
  assert np.unique(my).shape == (linear_ids.shape[0],)  # no repeat matches
  return my # index ids (2nd) stuff with


def add_non_imaging_fields(all_vars):
  # bb_flat, bb_flat_age, blood_flat, blood_flat_age : n, num_fields
  # BB
  bb_flat = []
  bb_flat_age = []
  bb_curr = all_vars["bb"]
  curr_bbi = 0
  for bbf, bb_shape in BB_fields: # ordered
    if bb_shape is not None:
      curr_bbi_end = curr_bbi + bb_shape
      if "age" in bbf:
        raise ValueError # age is always scalar
      else:
        if bbf in render_bb_vars:
          bbf_curr = bb_curr[:, curr_bbi:curr_bbi_end]
          bbf_flat_curr = bbf_curr[:, BB_scalar_inds[bbf]] # 1 if 1 in the specified discrete index, 0 otherwise
          assert len(bb_flat) == len(bb_flat_age)
          assert render_bb_vars[len(bb_flat)] == bbf
          bb_flat.append(bbf_flat_curr)
    else:
      curr_bbi_end = curr_bbi + 1
      if "age" in bbf:
        bbf_orig = bbf[4:]
        if bbf_orig in render_bb_vars:
          assert bbf[:4] == "age_"
          assert len(bb_flat_age) == len(bb_flat) - 1
          assert render_bb_vars[len(bb_flat) - 1] == bbf_orig # just added value to bb_flat
          bb_flat_age.append(bb_curr[:, curr_bbi:curr_bbi_end].squeeze()) # could also directly index curr_bbi
      else:
        if bbf in render_bb_vars:
          assert len(bb_flat) == len(bb_flat_age)
          assert render_bb_vars[len(bb_flat)] == bbf
          bb_flat.append(bb_curr[:, curr_bbi:curr_bbi_end].squeeze())

    curr_bbi = curr_bbi_end

  all_vars["bb_flat"] = np.stack(bb_flat, axis=1)
  all_vars["bb_flat_age"] = np.stack(bb_flat_age, axis=1)

  # BLOOD
  blood_ind = []
  for bf in render_blood_vars:
    blood_ind.append(BLOOD_fields.index((bf, None)))
  blood_ind = np.array(blood_ind)

  all_vars["blood_flat"] = all_vars["blood"][:, blood_ind]
  all_vars["blood_flat_age"] = all_vars["blood"][:, blood_ind + 1]
  print("added to all_vars shape", all_vars["bb_flat"].shape, all_vars["bb_flat_age"].shape, all_vars["blood_flat"].shape, all_vars["blood_flat_age"].shape)


def print_latex(list_list, ncol=5):
  print("----------- print_latex -------------")
  for l in list_list:
    l = [l[i].replace("/", "/ ") for i in range(len(l))]
    print(l[0] + " & " + ", ".join(l[1:(ncol+1)]) + " \\\\")
  print("----------- print_latex -------------")
  sys.stdout.flush()