import numpy as np
import Bert_preprocessing
import matplotlib.pyplot as plt

#load word vector df
sentence_vecs = np.load("sentence_vectors_pca.npy")

def cosine_dis(p1, p2):
    cos_sim = np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))
    cos_dis = 1 - cos_sim
    return cos_dis

#clustering Kmeans
#Through the test, the result of clustering between K-means and K-means++ has no significant differences, so this project simply uses K-means
#Clustering
def clustering(cens):
    clusters = [[] for _ in range(K)]

    for i in range(len(sentence_vecs)):
        min_dis = float('inf')
        close_cluster = -1

        for k in range(K):
            cur_dis = cosine_dis(sentence_vecs[i],cens[k])
            if cur_dis < min_dis:
                min_dis = cur_dis
                close_cluster = k

        clusters[close_cluster].append(i)

    return clusters

#Recalculate cluster center for each cluster
def re_centroids(clusters):
    new_centroids = []
    for cluster in clusters:
        #Calculate the mean
        cluster_vecs = sentence_vecs[cluster]
        new_cen = np.mean(cluster_vecs,axis=0)
        #Normalize the center
        new_cen = new_cen / np.linalg.norm(new_cen)
        new_centroids.append(new_cen)

    return new_centroids

def p_silhouette(p_idx,labels,clustering_res,K):
    if K == 1:
        return 0

    cluster_idx = labels[p_idx]
    inter_dis = 0
    nearest_dis = float('inf')

    #Inter distance
    for idx in clustering_res[cluster_idx]:
        if idx != p_idx:
            cos_dis = cosine_dis(sentence_vecs[p_idx],sentence_vecs[idx])
            inter_dis += cos_dis
    inter_dis /= (len(clustering_res[cluster_idx])-1)

    #The distance to the nearest cluster
    for c_idx in range(len(clustering_res)):
        if c_idx != cluster_idx:
            outer_dis = 0
            for idx in clustering_res[c_idx]:
                cos_dis = cosine_dis(sentence_vecs[p_idx],sentence_vecs[idx])
                outer_dis += cos_dis

            outer_dis /= len(clustering_res[c_idx])
            if outer_dis < nearest_dis:
                nearest_dis = outer_dis

    p_silhouette_score = (nearest_dis - inter_dis) / max(nearest_dis,inter_dis)
    return p_silhouette_score

def SSE(clusetering_res,re_centroids_res):
    sse = 0
    for i in range(len(clusetering_res)):
        p_indicis = clusetering_res[i]
        centroid = re_centroids_res[i]
        for p in p_indicis:
            l1_dis = sentence_vecs[p] - centroid
            sse += np.linalg.norm(l1_dis)**2

    return sse

def Kmeans(K):
    # Randomly select K initial centroids
    np.random.seed(93)
    centroids_indices = np.random.choice(len(sentence_vecs),K,replace=False)
    centroids = sentence_vecs[centroids_indices]

    #first clustering
    clustering_res = clustering(centroids)

    #first re centroids
    re_centroids_res = re_centroids(clustering_res)

    #Iteration 100 times
    max_iter = 100
    for i in range(max_iter):
        clustering_res = clustering(re_centroids_res)
        re_centroids_res = re_centroids(clustering_res)

    #Generate labels
    labels = {}
    for i in range(len(clustering_res)):
        cluster = clustering_res[i]
        for point_idx in cluster:
            labels[point_idx] = i
    labels = dict(sorted(labels.items()))

    #silhouette coefficient
    sum_silhouette = 0
    for i in range(len(sentence_vecs)):
        sum_silhouette += p_silhouette(i,labels,clustering_res,K)

    silhouette_score = sum_silhouette / len(sentence_vecs)

    # SSE(Sum of Squared Errors)
    sse = SSE(clustering_res,re_centroids_res)

    print(f"K = {K} | silhouette score:{silhouette_score:.4f} | SSE:{sse:.4f}")

    return silhouette_score,sse,labels

#Generate Silhouette analysis plot
silhouette_scores = {}
labels = []
SSEs = []
for K in range(1,11):
    s_score,sse,label = Kmeans(K)
    silhouette_scores[K] = s_score
    SSEs.append(sse)
    labels.append(label)

k = list(silhouette_scores.keys())
s_scores = list(silhouette_scores.values())

#Result visualization
plt.figure()
#Silhouette coefficient
plt.plot(k,s_scores)
plt.xlabel("K")
plt.ylabel("Silhouette coefficient")
plt.title("Silhouette coefficient in different K")
plt.grid(True)

plt.xticks(range(1, 11))
plt.savefig("Silhouette_Analysis_Plot.png", dpi=300, bbox_inches='tight')

best_Silhouette = max(silhouette_scores, key=silhouette_scores.get)
print(f"Silhouette best K: {best_Silhouette}")

#SSE
plt.plot(k,SSEs)
plt.xlabel("K")
plt.ylabel("SSE")
plt.title("SSE in different K")
plt.grid(True)

plt.xticks(range(1, 11))
plt.savefig("SSE.png", dpi=300, bbox_inches='tight')

#Find the elbow point of the SSE graph.
first_diff = np.diff(SSEs)
second_diff = np.diff(first_diff)
elbow_point = np.argmax(second_diff) + 3
print(f"Elbow K in SSE: {elbow_point}")


#Output the labels to a txt file.
if best_Silhouette == elbow_point:
    with open("labels.txt", "w") as f:
        for v in labels[best_Silhouette-1].values():
            f.write(f"{v+1}\n")
else:
    with open("Silhouette_labels.txt", "w") as f:
        for v in labels[best_Silhouette-1].values():
            f.write(f"{v+1}\n")
    with open("SSE_labels.txt", "w") as f:
        for v in labels[elbow_point-1].values():
            f.write(f"{v+1}\n")
