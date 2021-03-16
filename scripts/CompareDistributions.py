import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr, skew, ttest_1samp, wilcoxon, ks_2samp, ranksums, mannwhitneyu
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

##Load all data 
data = pd.read_csv(r'C:\Users\nandy\Downloads\energy_harvesters_vector.csv',index_col=0, header=2) #Load just numerical information
data2 = pd.read_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\02_intermediate\appliances_and_toys_vector_V2.csv',index_col=0, header=1).sort_index()


hamming_sim_full2 = pd.DataFrame(squareform(pdist(data2, metric='hamming')), columns = data2.index, index = data2.index)
jaccard_sim_full2 = pd.DataFrame(squareform(pdist(data2, metric='jaccard')), columns = data2.index, index = data2.index)
cosine_sim_full2 = pd.DataFrame(squareform(pdist(data2, metric='cosine')), columns = data2.index, index = data2.index)

hamming_sim_full = pd.DataFrame(squareform(pdist(data, metric='hamming')), columns = data.index, index = data.index)
jaccard_sim_full = pd.DataFrame(squareform(pdist(data, metric='jaccard')), columns = data.index, index = data.index)
cosine_sim_full = pd.DataFrame(squareform(pdist(data, metric='cosine')), columns = data.index, index = data.index)

networkdata = pd.read_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\03_processed\energy_harvesters_lambdadistance_sorted.csv',index_col=0, header=0)
networkdata2 = pd.read_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\03_processed\appliances_and_toys_lambdadistance_V2.csv',index_col=0, header=0)

networkdata_deltacon = pd.read_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\03_processed\energy_harvesters_deltacon_sorted.csv',index_col=0, header=0)
networkdata_deltacon2 = pd.read_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\03_processed\appliances_and_toys_deltacon_V2.csv',index_col=0, header=0)

networkdata_netsimile = pd.read_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\03_processed\energy_harvesters_netsimile.csv',index_col=0, header=0)
networkdata_netsimile2 = pd.read_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\03_processed\appliances_and_toys_netsimile_V2.csv',index_col=0, header=0)

##Range normalize and turn from distance to similarity
hamming_sim_full_norm = 1 - (hamming_sim_full - hamming_sim_full.min().min())/(hamming_sim_full.max().max() - hamming_sim_full.min().min())
jaccard_sim_full_norm = 1 - (jaccard_sim_full - jaccard_sim_full.min().min())/(jaccard_sim_full.max().max() - jaccard_sim_full.min().min())
cosine_sim_full_norm = 1 - (cosine_sim_full - cosine_sim_full.min().min())/(cosine_sim_full.max().max() - cosine_sim_full.min().min())
lambdadist_norm = 1 - ((networkdata - networkdata.min().min())/(networkdata.max().max() - networkdata.min().min()))
deltacon_norm = 1 - ((networkdata_deltacon - networkdata_deltacon.min().min())/(networkdata_deltacon.max().max() - networkdata_deltacon.min().min()))
netsimile_norm = 1 - ((networkdata_netsimile - networkdata_netsimile.min().min())/(networkdata_netsimile.max().max() - networkdata_netsimile.min().min()))

hamming_sim_full_norm2 = 1 - (hamming_sim_full2 - hamming_sim_full2.min().min())/(hamming_sim_full2.max().max() - hamming_sim_full2.min().min())
jaccard_sim_full_norm2 = 1 - (jaccard_sim_full2 - jaccard_sim_full2.min().min())/(jaccard_sim_full2.max().max() - jaccard_sim_full2.min().min())
cosine_sim_full_norm2 = 1 - (cosine_sim_full2 - cosine_sim_full2.min().min())/(cosine_sim_full2.max().max() - cosine_sim_full2.min().min())
lambdadist_norm2 = 1 - ((networkdata2 - networkdata2.min().min())/(networkdata2.max().max() - networkdata2.min().min()))
deltacon_norm2 = 1 - ((networkdata_deltacon2 - networkdata_deltacon2.min().min())/(networkdata_deltacon2.max().max() - networkdata_deltacon2.min().min()))
netsimile_norm2 = 1 - ((networkdata_netsimile2 - networkdata_netsimile2.min().min())/(networkdata_netsimile2.max().max() - networkdata_netsimile2.min().min()))

##Function to get upper triangle with no diagonal
def triu_nodiag(df):
    new = df.where(np.triu(np.ones(df.shape),k=1).astype(np.bool))
    print(new.max().max())
    print(new.max(axis=0).idxmax())
    print(new.max(axis=1).idxmax())
    new_flat = new.to_numpy().flatten()
    final = new_flat[~np.isnan(new_flat)]
    print(final[(-final).argsort()[:1]])
    return final


#Histograms for each measure
def simhistogram(normsim, measurename):
    #d1 = normsim.where(np.triu(np.ones(normsim.shape)).astype(np.bool)) #normalized upper triangle
    d1 = triu_nodiag(normsim) 
    #d1 = normsim
    #d1 = d1.to_numpy().flatten()
    d1 = d1.flatten()
    d1 = d1[~np.isnan(d1)] #A[~np.isnan(A)]
    plt.hist(d1)
    plt.grid(axis='y')
    plt.title(measurename)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.axvline(d1.mean(), color='k', linewidth=1)
    #plt.axvline(np.median(d1), color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(d1.mean(), max_ylim*0.9, 'Mean: {:.2f}'.format(d1.mean()))
    print(skew(d1))
    plt.show()
    return d1

p1 = simhistogram(hamming_sim_full_norm, 'Hamming')
p2 = simhistogram(jaccard_sim_full_norm, 'Jaccard')
p3 = simhistogram(cosine_sim_full_norm, 'Cosine')
p5 = simhistogram(lambdadist_norm, 'Spectral')
p6 = simhistogram(deltacon_norm, 'DeltaCon')
p9 = simhistogram(netsimile_norm, 'NetSimile')

q1 = simhistogram(hamming_sim_full_norm2, 'Hamming')
q2 = simhistogram(jaccard_sim_full_norm2, 'Jaccard')
q3 = simhistogram(cosine_sim_full_norm2, 'Cosine')
q5 = simhistogram(lambdadist_norm2, 'Spectral')
q6 = simhistogram(deltacon_norm2, 'DeltaCon')
q9 = simhistogram(netsimile_norm2, 'NetSimile')

print(ranksums(p1,q1))
print(ranksums(p2,q2))
print(ranksums(p3,q3))
print(ranksums(p5,q5))
print(ranksums(p6,q6))
print(ranksums(p9,q9))

print(mannwhitneyu(p1,q1))
print(mannwhitneyu(p2,q2))
print(mannwhitneyu(p3,q3))
print(mannwhitneyu(p5,q5))
print(mannwhitneyu(p6,q6))
print(mannwhitneyu(p9,q9))