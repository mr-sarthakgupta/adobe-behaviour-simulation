import os
import pickle
from tqdm import tqdm
import numpy as np
import faiss
from faiss import write_index, read_index

def load_image_tweet_embeddings():
    image_embeddings = []
    image_ids = []

    for filename in os.listdir('image_tweet_embed'):
        if filename.endswith('.pkl') and filename.startswith('image_'):
            with open(f'image_tweet_embed/{filename}', "rb") as fIn:
                data = pickle.load(fIn)
                image_embeddings.extend(data['embeddings'])
                image_ids.extend(data['ids'])

    text_embeddings = []
    text_ids = []

    with open('image_tweet_embed/text.pkl', "rb") as fIn:
        data = pickle.load(fIn)
        text_embeddings.extend(data['embeddings'])
        text_ids.extend(data['ids'])

    likes = []
    like_ids = []

    with open('image_tweet_embed/likes.pkl', "rb") as fIn:
        data = pickle.load(fIn)
        likes.extend(data['embeddings'])
        like_ids.extend(data['ids'])

    text_embeddings_new = []
    likes_new = []

    for i, id in tqdm(enumerate(image_ids)):
        text_embeddings_new.append(text_embeddings[text_ids.index(id)])
        likes_new.append(likes[like_ids.index(id)])

    assert len(text_embeddings_new) == len(likes_new) == len(image_embeddings)

    image_embeddings = np.array(image_embeddings)
    text_embeddings_new = np.array(text_embeddings_new)
    likes_new = np.array(likes_new)

    return image_embeddings, text_embeddings_new, likes_new

def load_video_tweet_embeddings():
    pass # TODO

def load_gif_tweet_embeddings():
    pass # TODO

if __name__ == '__main__':
    image_embeddings, image_text_embeddings, image_likes = load_image_tweet_embeddings()
    video_embeddings, video_text_embeddings, video_likes = load_video_tweet_embeddings()
    gif_embeddings, gif_text_embeddings, gif_likes = load_gif_tweet_embeddings()

    # text_embeddings = np.concatenate((image_text_embeddings, video_text_embeddings, gif_text_embeddings), axis=0)
    # media_embeddings = np.concatenate((image_embeddings, video_embeddings, gif_embeddings), axis=0)
    # likes = np.concatenate((image_likes, video_likes, gif_likes), axis=0)

    ## until video and gif are implemented as single pipeline
    text_embeddings = image_text_embeddings
    media_embeddings = image_embeddings
    likes = image_likes
    ######################################

    multimodal_embeddings = np.concatenate((media_embeddings, text_embeddings), axis=1)

    _, embedding_dim = multimodal_embeddings.shape

    train_size = int(0.8 * len(multimodal_embeddings))
    test_size = len(multimodal_embeddings) - train_size

    train_indices = np.random.choice(len(multimodal_embeddings), train_size, replace=False)
    test_indices = np.setdiff1d(np.arange(len(multimodal_embeddings)), train_indices)

    train_multimodal_embeddings = multimodal_embeddings[train_indices]
    test_multimodal_embeddings = multimodal_embeddings[test_indices]
    train_likes = np.array(likes)[train_indices]
    test_likes = np.array(likes)[test_indices]

    nlist = 50
    quantizer = faiss.IndexFlatL2(embedding_dim)
    index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)

    index.train(train_multimodal_embeddings)
    index.add(train_multimodal_embeddings)

    K = 15
    sq_error = []

    def remove_outliers(arr, weights, factor=1):
        q1 = np.percentile(arr, 20)
        q3 = np.percentile(arr, 80)

        iqr = q3 - q1

        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr

        filtered_indices = np.where((arr >= lower_bound) & (arr <= upper_bound))[0]

        filtered_arr = arr[filtered_indices]
        filtered_weights = weights[filtered_indices]

        return filtered_arr, filtered_weights

    print('testing on 20% of train data')

    for i, embedding in enumerate(test_multimodal_embeddings):
        D, I = index.search(np.array([embedding]), K)

        nearest_likes = []

        for id in I[0]:
            nearest_likes.append(train_likes[id])
        
        nearest_likes, nearest_weights = remove_outliers(np.array(nearest_likes), weights=D[0])

        try:
            predicted_likes = np.average(nearest_likes, weights=nearest_weights)
        except ZeroDivisionError:
            predicted_likes = sum(nearest_likes) / len(nearest_likes)

        actual_likes = test_likes[i]

        sq_error.append((predicted_likes - actual_likes) ** 2)

        if i % 1000 == 0:
            print(f'rmse after {i} samples:', np.sqrt(np.mean(sq_error)))

    print('storing index for full train data')

    index_full = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
    index_full.train(multimodal_embeddings)
    index_full.add(multimodal_embeddings)

    write_index(index_full, 'multimodal-index.index')
