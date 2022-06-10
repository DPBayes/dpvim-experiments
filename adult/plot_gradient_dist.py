import pandas as pd
import numpy as np
import argparse
from typing import List, Dict
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import neurips_fig_style
plt.rcParams.update(neurips_fig_style)

parser = argparse.ArgumentParser(fromfile_prefix_chars="%")
parser.add_argument("variant", choices=('ng', 'vanilla', 'aligned', 'aligned_ng', 'precon'))
parser.add_argument("--epsilon", type=float, default=None, help="Privacy level")
parser.add_argument("--dp_scale", type=float, default=None, help="DP-SGD noise std")
parser.add_argument("--num_epochs", "-e", default=4000, type=int, help="Number of training epochs.")
parser.add_argument("--sampling_ratio", "-q", default=0.01, type=float, help="Subsampling ratio for DP-SGD.")
parser.add_argument("--store_gradients", action="store_true", default=False)
parser.add_argument("--normalize", type=int, default=1)
parser.add_argument("--output_path", type=str, default="../results/", help="Path to folder in which to store parameter traces; defaults to '../results'")
parser.add_argument("--init_auto_scale", type=float, default=0.1, help="Initial std for the VI posterior")
parser.add_argument("--minmax_normalize", action="store_true", default=False)
parser.add_argument("--adjusted_regression", action="store_true", default=False)
parser.add_argument("--optim", type=str, default="adam", help="Gradient optimizer")
args, unknown_args = parser.parse_known_args()


if args.minmax_normalize:
    C_ng = 0.1 # chosen by hand
    C_vanilla = 2.0 # chosen by hand
    C_aligned = 2.0 # chosen by hand
    C_aligned_ng = 0.1 # chosen by hand
    C_precon = 3.0 # chosen by hand
else:
    C_ng = 0.1 # chosen by hand
    C_vanilla = 2.0 # chosen by hand
    C_aligned = 2.0 # chosen by hand
    C_aligned_ng = 0.01 # chosen by hand
    C_precon = 3.0 # chosen by hand

Cs = {
    'ng': C_ng,
    'vanilla': C_vanilla,
    'aligned': C_aligned,
    'aligned_ng': C_aligned_ng,
    'precon': C_precon
}

if args.epsilon is None and args.dp_scale is not None:
    dp_name_part = f"dp_scale{args.dp_scale}"
elif args.epsilon is not None:
    dp_name_part = f"eps{args.epsilon}"


seeds = range(123, 123+20)

q = args.sampling_ratio

######
## filename template
def filenamer(variant, suffix, C, args=args):
    dp_name_part = f"eps{args.epsilon}"
    if args.optim == "adam":
        return f"{args.output_path}/adult_{variant}_ne{args.num_epochs}_C{C}_q{q}_{dp_name_part}_auto_scale{args.init_auto_scale}_seed{args.seed}_minmax{args.minmax_normalize}_optim{args.optim}_{suffix}"
    elif args.optim == "sgd":
        return f"{args.output_path}/adult_{variant}_ne{args.num_epochs}_C{C}_q{q}_{dp_name_part}_auto_scale{args.init_auto_scale}_seed{args.seed}_minmax{args.minmax_normalize}_optim{args.optim}_lr{args.sgd_lr}_{suffix}"
##

plot_df = pd.DataFrame()
site_cats = pd.CategoricalDtype(['auto_loc', 'auto_scale'])

for seed in tqdm(seeds):
    args.seed = seed
    grad_traces = pd.read_pickle(filenamer(args.variant, "grad_trace", Cs[args.variant]) + ".p") # type: List[Dict[str, np.ndarray]]
    for grads in grad_traces[:1]:
        for site_name, site_grads in grads.items():
            seed_df = pd.DataFrame({
                'px_grad_norm': np.linalg.norm(site_grads, axis=-1),
                'site': site_name,
            })
            seed_df['site'] = seed_df['site'].astype(site_cats)
            plot_df = pd.concat((plot_df, seed_df))
            # plot_df = pd.concat((plot_df,
            #     pd.DataFrame({
            #             'px_grad_norm': np.linalg.norm(site_grads, axis=-1),
            #             'site': site_name,
            #         },
            #         dtype = {
            #             'px_grad_norm': np.float32,
            #             'site': 'category'
            #         }
            #     )
            # ))

id_fn = lambda x: x
transform_fn = np.log # id_fn
fig, ax = plt.subplots()
bins = np.histogram_bin_edges(transform_fn(plot_df['px_grad_norm']), bins=100)
for site in ['auto_loc', 'auto_scale']:
    ax.hist(transform_fn(plot_df[plot_df['site'] == site]['px_grad_norm']), bins=bins, alpha=.8, label=site)

ax.set_xlabel("$\log(||g||_2)")
fig.suptitle(f'Distribution of gradient norms, $q_s =${args.init_auto_scale}')
# ax = sns.histplot(data=plot_df, x='px_grad_norm', hue='site')
plt.savefig(f"grad_norm_dist_{args.variant}_{dp_name_part}_minmax{args.minmax_normalize}.pdf", format="pdf")
plt.close()