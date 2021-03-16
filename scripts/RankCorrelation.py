import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr, skew, ttest_1samp, wilcoxon 
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

##Load all data 
data = pd.read_csv(r'C:\Users\nandy\Downloads\energy_harvesters_vector.csv',index_col=0, header=2) #Load just numerical information
#data = pd.read_csv(r'C:\Users\nandy\Downloads\energy_harvesters_functionsandflows.csv',index_col=0, header=0) #data from Weaver method
#data = pd.read_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\02_intermediate\appliances_and_toys_vector_V2.csv',index_col=0, header=1).sort_index()
#data = pd.read_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\02_intermediate\combined_vector.csv',index_col=0, header=1)
#data = pd.read_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\02_intermediate\limitedcombined_adjacencymatrix.csv',index_col=0, header=1)

hamming_sim_full = pd.DataFrame(squareform(pdist(data, metric='hamming')), columns = data.index, index = data.index)
jaccard_sim_full = pd.DataFrame(squareform(pdist(data, metric='jaccard')), columns = data.index, index = data.index)
cosine_sim_full = pd.DataFrame(squareform(pdist(data, metric='cosine')), columns = data.index, index = data.index)
euc_sim_full = pd.DataFrame(squareform(pdist(data, metric='euclidean')), columns = data.index, index = data.index)
#hamming_sim_full.to_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\03_processed\appliances_and_toys_hammingdist_V2.csv')
#jaccard_sim_full.to_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\03_processed\appliances_and_toys_jaccarddist_V2.csv')
#cosine_sim_full.to_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\03_processed\appliances_and_toys_cosinedist_V2.csv')

#networkdata = pd.read_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\03_processed\energy_harvesters_lambdadistance.csv',index_col=0, header=0)
networkdata = pd.read_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\03_processed\energy_harvesters_lambdadistance_sorted.csv',index_col=0, header=0)
#networkdata = pd.read_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\03_processed\appliances_and_toys_lambdadistance_V2.csv',index_col=0, header=0)
#networkdata = pd.read_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\03_processed\combined_lambdadistance.csv',index_col=0, header=0)
#networkdata = pd.read_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\03_processed\limitedcombined_lambdadistance.csv',index_col=0, header=0)


#networkdata_ged = pd.read_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\03_processed\energy_harvesters_geddistance.csv',index_col=0, header=0)
networkdata_ged = pd.read_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\03_processed\energy_harvesters_geddistance_sorted.csv',index_col=0, header=0)
#networkdata_ged = pd.read_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\03_processed\appliances_and_toys_geddistance_V2.csv',index_col=0, header=0)
#networkdata_ged = pd.read_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\03_processed\combined_geddistance.csv',index_col=0, header=0)
#networkdata_ged = pd.read_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\03_processed\limitedcombined_geddistance.csv',index_col=0, header=0)


#networkdata_deltacon = pd.read_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\03_processed\energy_harvesters_deltacon.csv',index_col=0, header=0)
networkdata_deltacon = pd.read_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\03_processed\energy_harvesters_deltacon_sorted.csv',index_col=0, header=0)
#networkdata_deltacon = pd.read_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\03_processed\appliances_and_toys_deltacon_V2.csv',index_col=0, header=0)
#networkdata_deltacon = pd.read_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\03_processed\combined_deltacon.csv',index_col=0, header=0)
#networkdata_deltacon = pd.read_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\03_processed\limitedcombined_deltacon.csv',index_col=0, header=0)

networkdata_veo = pd.read_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\03_processed\energy_harvesters_vertexedgeoverlap.csv',index_col=0, header=0)
networkdata_resistance = pd.read_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\03_processed\energy_harvesters_resistance.csv',index_col=0, header=0)
networkdata_netsimile = pd.read_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\03_processed\energy_harvesters_netsimile.csv',index_col=0, header=0)
#networkdata_veo = pd.read_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\03_processed\appliances_and_toys_vertexedgeoverlap_V2.csv',index_col=0, header=0)
#networkdata_resistance = pd.read_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\03_processed\appliances_and_toys_resistance_V2.csv',index_col=0, header=0)
#networkdata_netsimile = pd.read_csv(r'C:\Users\nandy\source\repos\FunctionalModelSimilarity\FunctionalModelSimilarity\data\03_processed\appliances_and_toys_netsimile_V2.csv',index_col=0, header=0)

##Range normalize and turn from distance to similarity
euc_sim_full_norm = 1 - (euc_sim_full - euc_sim_full.min().min())/(euc_sim_full.max().max() - euc_sim_full.min().min())
hamming_sim_full_norm = 1 - (hamming_sim_full - hamming_sim_full.min().min())/(hamming_sim_full.max().max() - hamming_sim_full.min().min())
jaccard_sim_full_norm = 1 - (jaccard_sim_full - jaccard_sim_full.min().min())/(jaccard_sim_full.max().max() - jaccard_sim_full.min().min())
cosine_sim_full_norm = 1 - (cosine_sim_full - cosine_sim_full.min().min())/(cosine_sim_full.max().max() - cosine_sim_full.min().min())
lambdadist_norm = 1 - ((networkdata - networkdata.min().min())/(networkdata.max().max() - networkdata.min().min()))
ged_norm = 1 - ((networkdata_ged - networkdata_ged.min().min())/(networkdata_ged.max().max() - networkdata_ged.min().min()))
deltacon_norm = 1 - ((networkdata_deltacon - networkdata_deltacon.min().min())/(networkdata_deltacon.max().max() - networkdata_deltacon.min().min()))
veo_norm = 1 - ((networkdata_veo - networkdata_veo.min().min())/(networkdata_veo.max().max() - networkdata_veo.min().min()))
resistance_norm = 1 - ((networkdata_resistance - networkdata_resistance.min().min())/(networkdata_resistance.max().max() - networkdata_resistance.min().min()))
netsimile_norm = 1 - ((networkdata_netsimile - networkdata_netsimile.min().min())/(networkdata_netsimile.max().max() - networkdata_netsimile.min().min()))

print(netsimile_norm)

##Function to get upper triangle with no diagonal
def triu_nodiag(df):
    new = df.where(np.triu(np.ones(df.shape),k=1).astype(np.bool))
    print(new.max().max())
    print(new.max(axis=0).idxmax())
    print(new.max(axis=1).idxmax())
    #print(new.min().min())
    #print(new.min(axis=0).idxmin())
    #print(new.min(axis=1).idxmin())
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
#p4 = simhistogram(euc_sim_full_norm, 'Euclidean')
p4 = simhistogram(ged_norm, 'GED')
p5 = simhistogram(lambdadist_norm, 'Spectral')
p6 = simhistogram(deltacon_norm, 'DeltaCon')
p7 = simhistogram(resistance_norm, 'Resistance')
#p8 = simhistogram(veo_norm, 'VEO')
p9 = simhistogram(netsimile_norm, 'NetSimile')


#KDE plot
width  = 3.5625
height = width / 1.618
testfig = plt.figure(figsize=(width,height))
sns.set_context("paper")
sns.set_palette(sns.dark_palette((260, 75, 60), reverse=True, input="husl"))
measurespread = sns.distplot(p1,  hist=False, kde=True, kde_kws={'linewidth':4,"shade": True}, label='SMC')
measurespread = sns.distplot(p2,  hist=False, kde=True, kde_kws={'linewidth':2,"shade": True}, label='Jaccard')
measurespread = sns.distplot(p3,  hist=False, kde=True, kde_kws={'linewidth':0.5,"shade": True}, label='Cosine')
measurespread = sns.distplot(p9,  hist=False, kde=True, kde_kws={'linestyle':"--","shade": True}, label='NetSimile')
#measurespread = sns.distplot(p4,  hist=False, kde=True, kde_kws={'linestyle':"--","shade": True}, label='GED')
#measurespread = sns.distplot(p4,  hist=False, kde=True, kde_kws={'linestyle':"--","shade": True}, label='Euclidean')
measurespread = sns.distplot(p5,  hist=False, kde=True, kde_kws={'linestyle':":","shade": True}, label='Spectral')
measurespread = sns.distplot(p6,  hist=False, kde=True, kde_kws={'linestyle':"-.","shade": True}, label='DeltaCon')
#measurespread = sns.distplot(p7,  hist=False, kde=True, kde_kws={'linestyle':"--","shade": True}, label='Resistance')
#measurespread = sns.distplot(p8,  hist=False, kde=True, kde_kws={'linestyle':":","shade": True}, label='VEO')
plt.xlim(-0.1,1.1)
plt.ylim(0,4)
plt.legend(loc='upper right',bbox_to_anchor=(1.53, 1.05)) #fix so legend automatically goes outside
plt.xlabel('Similarity Score')
plt.ylabel('Probability Density')
plt.subplots_adjust(right=0.7,bottom=0.25)
#plt.ylim([0,25])
plt.show()
testfig.savefig('DistributionsV3_EnergyHarvesters.pdf', dpi=300)

##Get rankings for each product separately (Kendall rank correlation)
def productrankingcorr(normsim1, normsim2, title):
    corrs = []
    sorted_1 = normsim1#.rank()
    sorted_2 = normsim2#.rank()
    #sorted_1 = normsim1.apply(np.argsort, axis=0) # get indices of each sorted row
    #sorted_2 = normsim2.apply(np.argsort, axis=0)
    for i in range(0,39):
        rankingcorr = sorted_1[sorted_1.columns[i]].corr(sorted_2[sorted_2.columns[i]], method='kendall')
        corrs.append(rankingcorr)
    n = len(corrs)
    reps = 500 #Number for bootstrapping
    np.random.seed(0)
    xb = np.random.choice(np.array(corrs), (n,reps)) #bootstrapping
    mb = xb.mean(axis=0)
    confint = np.percentile(mb, [2.5, 97.5])
    #plt.hist(corrs)
    #plt.title(title)
    #plt.show()
    return  (corrs,np.round(confint,2),np.round(mb.mean(),2))


(d1,c1,m1) = productrankingcorr(jaccard_sim_full_norm, lambdadist_norm, 'Jaccard-Spectral rank corr')
(d2,c2,m2) = productrankingcorr(hamming_sim_full_norm, jaccard_sim_full_norm, 'Hamming-Jaccard rank corr')
(d3,c3,m3) = productrankingcorr(hamming_sim_full_norm, cosine_sim_full_norm, 'Hamming-Cosine rank corr')
(d4,c4,m4) = productrankingcorr(hamming_sim_full_norm, netsimile_norm, 'Hamming-NetSimile rank corr')
#(d4,c4,m4) = productrankingcorr(hamming_sim_full_norm, ged_norm, 'Hamming-GED rank corr')
(d5,c5,m5) = productrankingcorr(hamming_sim_full_norm, lambdadist_norm, 'Hamming-Spectral rank corr')
(d6,c6,m6) = productrankingcorr(hamming_sim_full_norm, deltacon_norm, 'Hamming-DeltaCon rank corr')
(d7,c7,m7) = productrankingcorr(jaccard_sim_full_norm, cosine_sim_full_norm, 'Jaccard-Cosine rank corr')
(d8,c8,m8) = productrankingcorr(jaccard_sim_full_norm, netsimile_norm, 'Jaccard-NetSimile rank corr')
#(d8,c8,m8) = productrankingcorr(jaccard_sim_full_norm, ged_norm, 'Jaccard-GED rank corr')
(d9,c9,m9) = productrankingcorr(jaccard_sim_full_norm, deltacon_norm, 'Jaccard-DeltaCon rank corr')
(d10,c10,m10) = productrankingcorr(cosine_sim_full_norm, netsimile_norm, 'Cosine-NetSimile rank corr')
#(d10,c10,m10) = productrankingcorr(cosine_sim_full_norm, ged_norm, 'Cosine-GED rank corr')
(d11,c11,m11) = productrankingcorr(cosine_sim_full_norm, lambdadist_norm, 'Cosine-Spectral rank corr')
(d12,c12,m12) = productrankingcorr(cosine_sim_full_norm, deltacon_norm, 'Cosine-DeltaCon rank corr')
(d13,c13,m13) = productrankingcorr(netsimile_norm, lambdadist_norm, 'NetSimile-Spectral rank corr')
#(d13,c13,m13) = productrankingcorr(ged_norm, lambdadist_norm, 'GED-Spectral rank corr')
(d14,c14,m14) = productrankingcorr(netsimile_norm, deltacon_norm, 'NetSimile-DeltaCon rank corr')
#(d14,c14,m14) = productrankingcorr(ged_norm, deltacon_norm, 'GED-DeltaCon rank corr')
(d15,c15,m15) = productrankingcorr(lambdadist_norm, deltacon_norm, 'Spectral-DeltaCon rank corr')

##Plot rank correlations
sns.set_palette(sns.dark_palette((260, 75, 60), input="husl"))
sns.set_context("paper")
# Set up the matplotlib figure
b = np.linspace(0, 1, 20)
f, axes = plt.subplots(5, 5, figsize=(7, 7), sharex=True, sharey=True)
width  = 7.5
height = width / 1.618
f.set_size_inches(width,height)
sns.despine()
sns.distplot(d2, kde=False, color="b", ax=axes[0, 0],bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
axes[0,0].set_ylabel('Jaccard')
axes[0,0].set_title("{} {}".format(m2,c2))
sns.distplot(d3, kde=False, color="b", ax=axes[1, 0],bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
axes[1,0].set_ylabel('Cosine')
axes[1,0].title.set_text("{} {}".format(m3,c3))
sns.distplot(d4, kde=False, color="b", ax=axes[2, 0], bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
axes[2,0].set_ylabel('NetSimile')
#axes[2,0].set_ylabel('Euclidean')
axes[2,0].title.set_text("{} {}".format(m4,c4))
sns.distplot(d5, kde=False, color="b", ax=axes[3, 0], bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
axes[3,0].set_ylabel('Spectral')
axes[3,0].title.set_text("{} {}".format(m5,c5))
sns.distplot(d6, kde=False, color="b", ax=axes[4, 0],bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
axes[4,0].set_ylabel('DeltaCon')
axes[4,0].set_xlabel('SMC')
axes[4,0].title.set_text("{} {}".format(m6,c6))
sns.distplot(d7, kde=False, color="b", ax=axes[1, 1],bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
axes[1,1].title.set_text("{} {}".format(m7,c7))
sns.distplot(d8, kde=False, color="b", ax=axes[2, 1],bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
axes[2,1].title.set_text("{} {}".format(m8,c8))
sns.distplot(d1, kde=False, color="b", ax=axes[3, 1],bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
axes[3,1].title.set_text("{} {}".format(m1,c1))
sns.distplot(d9, kde=False, color="b", ax=axes[4, 1],bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
axes[4,1].set_xlabel('Jaccard')
axes[4,1].title.set_text("{} {}".format(m9,c9))
sns.distplot(d10, kde=False, color="b", ax=axes[2, 2],bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
axes[2,2].title.set_text("{} {}".format(m10,c10))
sns.distplot(d11, kde=False, color="b", ax=axes[3, 2],bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
axes[3,2].title.set_text("{} {}".format(m11,c11))
sns.distplot(d12, kde=False, color="b", ax=axes[4, 2],bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
axes[4,2].set_xlabel('Cosine')
axes[4,2].title.set_text("{} {}".format(m12,c12))
sns.distplot(d13, kde=False, color="b", ax=axes[3, 3],bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
axes[3,3].title.set_text("{} {}".format(m13,c13))
sns.distplot(d14, kde=False, color="b", ax=axes[4, 3],bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
axes[4,3].set_xlabel('NetSimile')
axes[4,3].title.set_text("{} {}".format(m14,c14))
sns.distplot(d15, kde=False, color="b", ax=axes[4, 4],bins=b, hist_kws=dict(edgecolor="k", linewidth=0.25))
axes[4,4].title.set_text("{} {}".format(m15,c15))
axes[4,4].set_xlabel('Spectral')
for i in range(5):
    for j in range(5):
        if i<j:
            axes[i, j].axis('off')
plt.tight_layout()
plt.subplots_adjust(left=0.1,bottom=0.15)
f.text(0.5,0.02, "Similarity Score", ha="center", va="center")
f.text(0.015,0.5, "Frequency", ha="center", va="center", rotation=90)
plt.setp(axes, ylim=(0,25))
plt.show()
f.savefig('RankCorrJournalConsumerProducts.pdf', dpi=300)

