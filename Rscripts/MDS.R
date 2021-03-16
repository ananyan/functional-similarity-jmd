setwd("C:/Users/nandy/Desktop/human-design-similarity") 
dc <- read.csv("designrepo_24_deltacon.csv",row.names=1)
netsim <- read.csv("designrepo_24_netsimile.csv",row.names=1)
spectral <- read.csv("designrepo_24_lambdadistance.csv",row.names=1)
jaccard <- read.csv("designrepo_24_jaccarddistance.csv",row.names=1)
cosine <- read.csv("designrepo_24_cosinedistance.csv",row.names=1)
hamming <- read.csv("designrepo_24_hammingdistance.csv",row.names=1)


d <- cosine
# from https://www.statmethods.net/advstats/mds.html
fit <- cmdscale(d,eig=TRUE, k=2) # k is the number of dim
fit # view results

# plot solution
x <- fit$points[,1]
y <- fit$points[,2]
plot(x, y, xlab="Coordinate 1", ylab="Coordinate 2",
     main="Metric MDS", type="n")
text(x, y, labels = row.names(d), cex=.7) #removed col = tvcls

fit <- isoMDS(d, k=2) # k is the number of dim
fit # view results

# plot solution
x <- fit$points[,1]
y <- fit$points[,2]
plot(x, y, xlab="Coordinate 1", ylab="Coordinate 2",
     main="Nonmetric MDS", type="n")
text(x, y, labels = row.names(d), cex=.7) #removed col = tvcls

