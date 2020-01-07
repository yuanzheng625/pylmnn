from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

from pylmnn import LargeMarginNearestNeighbor
# raw data
loreal_path = '/export/home//loreal_135_classification/loreal_135.npz'
loreal_data = np.load(loreal_path)
X_all, y_all = loreal_data['X'], loreal_data['y']

X_all = X_all.reshape(X_all.shape[0], -1)

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.7, stratify=y_all)

np.random.seed(0)
rd_idx_train = np.random.choice([i for i in range(X_train.shape[0])], size=X_train.shape[0]//100)
rd_idx_test = np.random.choice([i for i in range(X_test.shape[0])], size=X_test.shape[0]//10)

X_train = X_train[rd_idx_train,:]
y_train = y_train[rd_idx_train]

X_test = X_test[rd_idx_test,:]
y_test = y_test[rd_idx_test]


'''
# embedding
loreal_data = np.load('/export/home//loreal_135_classification/em_training.npz')
X_train, y_train = loreal_data['X'], loreal_data['y']

loreal_data = np.load('/export/home//loreal_135_classification/em_test.npz')
X_test, y_test = loreal_data['X'], loreal_data['y']
'''

knn = KNeighborsClassifier(n_neighbors=10)

# Train with no transformation (euclidean metric)
knn.fit(X_train, y_train)

# Test with euclidean metric
acc = knn.score(X_test, y_test)

print('KNN accuracy on test set: {}'.format(acc))


# LMNN is no longer a classifier but a transformer
lmnn = LargeMarginNearestNeighbor(n_neighbors=10, verbose=1, max_iter=300)
lmnn.fit(X_train, y_train)

# Train with transformation learned by LMNN
knn.fit(lmnn.transform(X_train), y_train)

# Test with transformation learned by LMNN
acc = knn.score(lmnn.transform(X_test), y_test)

print('LMNN accuracy on test set: {}'.format(acc))
