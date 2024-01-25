import pandas as pd
import numpy as np
from tqdm import tqdm

df = pd.read_csv("train.csv")
# 过滤超长文本
sent1_cond = df["sentence1"].map(lambda x:len(x)<100)
sent2_cond = df["sentence2"].map(lambda x:len(x)<100)
df = df.loc[(sent1_cond & sent2_cond), ["sentence1", "sentence2", "label"]]

# 根据数据集获取Embedding
all_vecs=[]
all_labels=[]
with tqdm(total=df.shape[0]) as pbar:
    for sent1, sent2, label in df.itertuples(index=None):
        vec1 = get_glm_embedding(model, tokenizer, sent1).reshape([-1, 4096])
        vec2 = get_glm_embedding(model, tokenizer, sent2).reshape([-1, 4096])
        all_vecs.append((vec1, vec2))
        all_labels.append(label)
        pbar.update(1)


# 计算kernel和bias
def compute_kernel_bias(vecs, n_components=256):
    """
    n_components为PCA前n维特征
    """
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    return W[:, :n_components], -mu

kernel, bias = compute_kernel_bias(vecs)

def get_glm_embedding(text, device="cuda:1"):
    inputs = tokenizer([text], return_tensors="pt").to(device)
    resp = model.transformer(**inputs, output_hidden_states=True)
    y = resp.last_hidden_state
    y_mean = torch.mean(y, dim=0, keepdim=True)
    return y_mean.cpu().detach().numpy()


def transform_and_normalize(vecs, kernel=None, bias=None):

    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5








