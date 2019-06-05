import numpy as np
from tqdm import tqdm
import csv
import os

import torch
from torch.utils.data import dataset, dataloader
from torchvision import transforms

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from data import Market1501
from network import MGN
from utils.extract_feature import extract_feature
from opt import opt

import shutil

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    model = MGN()
    model.load_state_dict(torch.load(opt.weight))

    cuda_model = model.to("cuda")

    embeddings = None
    file_names = None

    cuda_model.eval()

    for dataset in opt.data_path: 

        test_transform = transforms.Compose([
            transforms.Resize((384, 128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        testset = Market1501(test_transform, 'test', dataset)
        data_loader = dataloader.DataLoader(testset, batch_size=16, num_workers=8, pin_memory=True)

        print("generating embeddings for", dataset)
        
        features = extract_feature(cuda_model, tqdm(data_loader)).numpy()
        names = np.array(testset.imgs)

        if embeddings is None:
            embeddings = features
            file_names = names
        else:
            embeddings = np.concatenate((embeddings, features))
            file_names = np.concatenate((file_names, names))


    if opt.embeddings_path is not None:
        np.save(opt.embeddings_path, embeddings)
        np.save(opt.embeddings_path + "_names", file_names)

    print("performing clustering")
    print("finding number of clusters")

    embeddings = StandardScaler().fit_transform(embeddings) 

    clusterings = {}
    scores = {}

    def get_score(x): 
        x = int(x)
        if x < 2:
            return 0
        if x not in clusterings:
            clusterings[x] = KMeans(x).fit(embeddings)
            scores[x] = silhouette_score(embeddings, clusterings[x].labels_)
            return scores[x]
        
        return scores[x]
        

    candidates = [2**x for x in range(1, 7)]
    candidate = candidates[np.argmax(np.array([get_score(a) for a in candidates]))]

    possible_ks = [x for x in range(candidate // 2, (candidate * 2) + 1)]

    k = possible_ks[np.argmax(np.array([get_score(a) for a in tqdm(possible_ks)]))]

    print("found best k", k, "with silhouette score of", scores[k])

    clustering = clusterings[k]
    
    with open(opt.output_file, "w") as csvfile:
        csv_writer = csv.writer(csvfile)

        for path, cluster_id, embedding in zip(file_names, clustering.labels_, embeddings):
            csv_writer.writerow([path, str(cluster_id)])
            if not os.path.isdir(str(cluster_id)):
                os.mkdir(str(cluster_id))
            shutil.copyfile(path, str(cluster_id) + "/" + path.split("\\")[-1])


if __name__ == "__main__":
    main()