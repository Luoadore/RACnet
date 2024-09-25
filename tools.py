import seaborn as sns
import matplotlib.pyplot as plt


class AverageMeter(object):
    "# source: https://github.com/HobbitLong/SupContrast/blob/master/util.py"
    def __init__(self, name: str, fmt: Optional[str] = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def set_seed(seed):
    torch.manual_seed(seed)                     # =torch.random.manual_seed
    torch.cuda.manual_seed_all(seed)            # w/o all for current GPU
    np.random.seed(seed)
    random.seed(seed)

def plot_action_start(data, save_path):
    plt.figure(figsize=(10, 2.7))
    x = np.arange(len(data))
    sns.lineplot(x=x, y=data)
    plt.savefig(save_path, dpi=50)
    print('plot curve to: ', save_path)

def plot_tsm_heatmap(heatmap, save_path):
    """
    use seaborn to plot the heatmap
    """
    plt.figure(figsize=(6, 6))
    cmap = "viridis"
    hm = sns.heatmap(data=heatmap, 
                cmap=cmap,
                xticklabels = False,
                yticklabels = False,
                square=True,
                cbar=False)
    plt.axis('off')
    plt.savefig(save_path)
    print('plot heatmap to: ', save_path)