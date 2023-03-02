import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import umap
import argparse

from sklearn.manifold import TSNE
import pandas as pd  
import pickle
import os

def draw_umap(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', data=None):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    u = fit.fit_transform(data)
    return u


def get_all_stats(PATH):
    """
    Args:
        PATH (str): Path to directory containing .pkl files
    Returns:
        y_error (np.array): Error between true and predicted affinity
        prot_seqs (str): List of Protein sequences
        prot_embed (np.array): Protein embeddings
    """
    # List all .pkl in PATH
    files = [f for f in os.listdir(PATH) if f.endswith('.pkl')]

    # Load all .pkl in PATH
    all_stats = []
    for f in files:
        with open(PATH + f, 'rb') as handle:
            all_stats.append(pickle.load(handle))

    y_pred = [x['y_pred'] for x in all_stats]
    y_true = [x['y_true'] for x in all_stats]
    prot_seqs = []
    for x in all_stats:
        for seq in x['protein_target_seqs']:
            prot_seqs.append(seq)
    prot_embed = [x['protein_embeddings'] for x in all_stats]

    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    y_error = (y_true - y_pred)**2
    prot_embed = np.concatenate(prot_embed)

    return y_error, prot_seqs, prot_embed

################## SCATTER PLOT ##################

def scatter_plot(all_model_stats, dataset_name):
    for model_stats in all_model_stats:

        y_error = all_model_stats[model_stats]['y_error']
        prot_lens = all_model_stats[model_stats]['prot_lens']
        prot_embed = all_model_stats[model_stats]['prot_embed']

        # Plot a dot for each protein
        plt.scatter(prot_lens, y_error, s=1)
        plt.xlabel('Protein Length')
        plt.ylabel('Test Error (Binding Affinity Difference)')
        plt.title(model_stats)

        # Save the plot to a file
        plt.savefig(PATH_TO_SAVE + str(model_stats) +  '_scatterplot.png')

        # clear the plot
        plt.clf()

        print("Total Error for {}: {}".format(model_stats, np.mean(y_error)))

# ################## POPULATION PLOT ##################

def population_plot(all_model_stats, dataset_name, PATH_TO_SAVE):
    for model_stats in all_model_stats:
        model = str(model_stats).replace('_', '').replace(dataset_name, '').replace(epochs, '').upper()
        print(model)

        y_error = all_model_stats[model_stats]['y_error']
        prot_lens = [ len(x) for x in all_model_stats[model_stats]['prot_seqs'] ]
        prot_embed = all_model_stats[model_stats]['prot_embed']

        min_population = min(prot_lens)
        max_population = max(prot_lens)

        min_index = int(min_population/200) * 200
        max_index = int(max_population/200 + 1) * 200

        population_values = [ str(x) + "-" + str(x + 200) for x in range(min_index, max_index, 200) ]
        population_mean_error = [[0, 0] for x in range(len(population_values))]
        for indx, prot_len in enumerate(prot_lens):
            population_mean_error[int((prot_len - min_index)/200)][0] += y_error[indx]
            population_mean_error[int((prot_len - min_index)/200)][1] += 1

        population_mean_error = [x[0]/x[1] if x[1] != 0 else 0 for x in population_mean_error]

        # Plot a histogram plot for each popluation value
        plt.bar(population_values, population_mean_error)
        plt.xlabel('Protein Length')
        plt.ylabel('Mean Test Error (Binding Affinity Difference)')
        plt.xticks(rotation = 90)
        plt.title(str(model) + ' (' + str(dataset_name).upper() + ')')
        plt.tight_layout()
        # Save the plot to a file
        plt.savefig(PATH_TO_SAVE + str(model_stats) + '_population_plot.png')
        plt.clf()


# ################## TSNE PLOT ##################

def plot_tsne(all_model_stats, dataset_name, PATH_TO_SAVE):
    for model_stats in all_model_stats:
        if 'cnn' in str(model_stats).lower():
            all_model_stats.pop(model_stats)
            break

    prot_embed_all = np.concatenate([all_model_stats[model_stats]['prot_embed'] for model_stats in all_model_stats])
    y_all = np.concatenate([[ str(model_stats) for i in range(len(all_model_stats[model_stats]['y_error'])) ] for model_stats in all_model_stats])

    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    prot_embed_transform = tsne.fit_transform(prot_embed_all)

    df = pd.DataFrame()
    df['y'] = y_all
    df["comp-1"] = prot_embed_transform[:,0]
    df["comp-2"] = prot_embed_transform[:,1]

    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", 3),
                    data=df).set(title="T-SNE projection")

    # fig = tsne_plot.get_figure()
    plt.savefig(PATH_TO_SAVE +  'tsne_' + str(dataset_name) + '_.png')
    plt.clf()

# ############# UMAP PLOT ####################

def umap_plot(all_model_stats, dataset_name, PATH_TO_SAVE):
    for model_stats in all_model_stats:
        if 'cnn' in str(model_stats).lower():
            all_model_stats.pop(model_stats)
            break

    prot_embed_all = np.concatenate([all_model_stats[model_stats]['prot_embed'] for model_stats in all_model_stats])
    y_all = np.concatenate([[ str(model_stats).replace('_', '').replace(dataset_name, '').replace('1500', '').upper() for i in range(len(all_model_stats[model_stats]['y_error'])) ] for model_stats in all_model_stats])

    prot_embed_transform = draw_umap(data=prot_embed_all)

    df = pd.DataFrame()
    df['y'] = y_all
    df["comp-1"] = prot_embed_transform[:,0]
    df["comp-2"] = prot_embed_transform[:,1]

    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", 3),
                    data=df).set(title="UMAP projection for " + str(dataset_name.upper()))

    plt.savefig(PATH_TO_SAVE + 'umap_' + str(dataset_name) + '.png')
    plt.clf()

def parse_args():
    # Argument Parser for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_choice', type=int, default=0, help='Dataset choice')
    parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':

    args = parse_args()

    datasets = ["DAVIS", "KIBA"]
    dataset_name = datasets[args.dataset_choice].lower()
    epochs = args.epochs
    models = ["SeqVec", "DistilProtBERT", "ProtBERT", "CNN"]
    models = [model.lower() for model in models]

    PATHS = ['./plots/' + model + '_' + epochs + '_' + dataset_name + '/' for model in models]

    all_model_stats = {}

    for PATH in PATHS:
        y_error, prot_seqs, prot_embed = get_all_stats(PATH)
        all_model_stats[PATH[2:-1]] = {'y_error': y_error, 'prot_seqs': prot_seqs, 'prot_embed': prot_embed}
        print(" - Loaded {} stats".format(PATH[2:-1]))

    PATH_TO_SAVE = './plots/created_plots/'
    if not os.path.exists(PATH_TO_SAVE):
        os.makedirs(PATH_TO_SAVE)

    scatter_plot(all_model_stats, dataset_name, PATH_TO_SAVE)
    population_plot(all_model_stats, dataset_name, PATH_TO_SAVE)
    plot_tsne(all_model_stats, dataset_name, PATH_TO_SAVE)
    umap_plot(all_model_stats, dataset_name, PATH_TO_SAVE)
