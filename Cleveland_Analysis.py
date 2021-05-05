"""

Data Science Project: Heart Disease Prediction

"""



import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import plotly.express as px
import matplotlib.pyplot as plt
import sklearn.model_selection as model_selection

from umap import UMAP

from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc

from keras.utils import to_categorical
from keras import optimizers, regularizers
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Input


# Load the dataset
df_c = pd.read_csv('cleveland_76_header.csv', sep=',')

df_c.head
df_c.columns
df_c.dtypes

# df_c.loc[(df_c.painloc == -9.0), 'painloc'] = 'NA'
# df_c['painloc']

# df_c[df_c == -9.0]

# Remove the missing values
cols = []
for i in df_c.columns:
    if df_c.loc[0, i] == -9:
        cols.append(i)
df_c[cols] = 'NA'
df_c = df_c.drop(cols, axis = 1)
df_c = df_c.drop('name', axis = 1)

# Vizualize the datasee
plt.scatter(x=df_c.age[df_c.num==1], y=df_c.thalach[(df_c.num==1)], c="indianred")
plt.scatter(x=df_c.age[df_c.num==2], y=df_c.thalach[(df_c.num==2)], c="indianred")
plt.scatter(x=df_c.age[df_c.num==3], y=df_c.thalach[(df_c.num==3)], c="indianred")
plt.scatter(x=df_c.age[df_c.num==4], y=df_c.thalach[(df_c.num==4)], c="indianred")
plt.scatter(x=df_c.age[df_c.num==0], y=df_c.thalach[(df_c.num==0)], c="lightskyblue")
plt.legend(["Disease", "Not Disease"], labelcolor = ["indianred", "lightskyblue"])
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
plt.show()

#  Representation of Cholestoral level
sns.scatterplot(data = df_c, x = "age", y = "chol", 
                hue = "sex", size = "chol", palette = "pastel")

# Comparison of blood pressure across pain type
df_c.loc[(df_c.cp == 1), 'cp'] = 'Typical angina'
df_c.loc[(df_c.cp == 2), 'cp'] = 'Atypical angina'
df_c.loc[(df_c.cp == 3), 'cp'] = 'Non-anginal pain'
df_c.loc[(df_c.cp == 4), 'cp'] = 'Asymptomatic'
sns.boxplot(data = df_c, x = "cp", y = "trestbps", 
            hue = "sex", palette = "pastel")

# Represent the correlation between the features and the target
# df_c.loc[(df_c.num == 1), 'num'] = 1
# df_c.loc[(df_c.num == 2), 'num'] = 1
# df_c.loc[(df_c.num == 3), 'num'] = 1
# df_c.loc[(df_c.num == 4), 'num'] = 1
X = df_c.drop(['num'], axis = 1)
X.corrwith(df_c['num']).plot.bar(figsize = (12, 6), 
                                 title = "Correlation with Target", fontsize = 10,
                                 rot = 90, grid = True)

# Split the dataset
X = df_c.loc[:, df_c.columns != 'num']
y = df_c.loc[:, 'num']
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, 
                                                                    train_size=0.8,
                                                                    test_size=0.2, 
                                                                    random_state=101)

#--------------------------------- Random Forests --------------------------------
# Measure the relative importance of each feature on the prediction
# Pros: scale invariant, robust to irrelevant features, interpretable
# Cons: tendency to overfit (not good for generalization)
random_forest = RandomForestClassifier(n_estimators = 100)
random_forest.fit(X_train, y_train)

Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
print(round(acc_random_forest,2,), "%")

importances = pd.DataFrame({'feature': X_train.columns,
                            'importance': np.round(random_forest.feature_importances_,3)})
importances = importances.sort_values('importance', ascending=False).set_index('feature')

importances.head(25)
importances.head(25).sum() # with 25 features, we can "explain" for 80 %

importances.plot.bar()
# the distal left anterior descending artery seems to be one of the most important features.
# Indeed, it is part of the left main coronary artery (LAD), considered the most important because it supplies more than half of the blood to the heart.


# Select the 25 most "important" features
ind = importances.head(25).index.tolist()
X = df_c.loc[:, df_c.columns != 'num']
X = X.loc[:, ind]
y = df_c.loc[:, 'num']

#X.to_csv(r'C:\Users\Kalvin\Desktop\Master\UZH\Data Science\X.csv')
#y.to_csv(r'C:\Users\Kalvin\Desktop\Master\UZH\Data Science\y.csv')


#--------------------------------- t-SNE --------------------------------
# Reduce the dimensionality of the dataset using a non-linear alternative to PCA (t-SNE)
plt.figure(figsize = (10,8))
for index, perp in enumerate([1, 10, 25, 75]):
    tsne = TSNE(n_components = 2, perplexity = perp, random_state = 0)
    embedded = pd.DataFrame(tsne.fit_transform(X), columns = ['tsne1', 'tsne2'])
    plt.subplot(2, 2, index + 1)
    plt.scatter(embedded['tsne1'], embedded['tsne2'], c = y.tolist(), s = 30)
    plt.title('Perplexity = {}'.format(str(perp)))
plt.show()
# Perplexity = 1 : local variations dominate
# Perplexity = 75 : global variations dominate

#--------------------------------- UMAP --------------------------------
# Reduce the dimensionality of the dataset using a non-linear alternative to PCA (UMAP)
umap_2d = UMAP(random_state=0, n_neighbors=15, min_dist=.15)
embedding = pd.DataFrame(umap_2d.fit_transform(X), columns = ['UMAP1','UMAP2'])
sns.scatterplot(x='UMAP1', y='UMAP2', data=embedding, 
                hue=y.tolist(),
                alpha=.9, linewidth=0.9)


#--------------------------------- Logistic Regression --------------------------------
# Use the logistic regression (try to merge 1, 2, 3, 4)
y.value_counts()
y.replace(['2', '3', '4'], '1')
y = y.astype(int)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, 
                                                                    train_size=0.8,
                                                                    test_size=0.2, 
                                                                    random_state=0)
log_regression = LogisticRegression()
log_regression.fit(X_train, y_train)
y_pred = log_regression.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

probs = log_regression.predict_proba(X_test)
probs = probs[:, 1]

auc = metrics.roc_auc_score(y_test, probs)

auc = metrics.roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)

fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)

plt.figure(figsize = (8,6))
plt.plot(fpr, tpr, color = 'orange', label = 'ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


#--------------------------------- Autoencoders --------------------------------
# https://github.com/georsara1/Autoencoders-for-dimensionality-reduction/blob/master/autoencoder.py
# Reduce the dimensionality of the data using a neural network approach
# Encode categorical variables to ONE-HOT
y_onehot = to_categorical(y)
#y_onehot = pd.get_dummies(y, prefix = 'y')

#Scale variables to [0,1] range
X.dtypes
X_scaled = X.apply(lambda x: (x-x.min())/(x.max()-x.min()))

#Split in 75% train and 25% test set
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_scaled, y_onehot2, 
                                                                    test_size = 0.15, 
                                                                    random_state = 0)

# Check distribution of labels in train and test set
X_train.shape
y_train.shape
X_test.shape
y_test.shape

# Build the Autoencoder
# Choose size of our encoded representations (we will reduce our initial features to this number)
encoding_dim = 2

# Define input layer
input_data = Input(shape = (X_train.shape[1],))

# Define encoding layer
encoded = Dense(encoding_dim, activation = 'relu')(input_data)

# Define decoding layer
decoded = Dense(X_train.shape[1], activation= 'sigmoid')(encoded)

# Create the autoencoder model
autoencoder = Model(input_data, decoded)

#Compile the autoencoder model
autoencoder.compile(optimizer='rmsprop', 
                    loss='sparse_categorical_crossentropy', 
                    metrics=['accuracy'])

autoencoder.compile(optimizer = 'adadelta', loss='categorical_crossentropy')

# Fit to train set, validate with dev set and save to hist_auto for plotting purposes
hist_auto = autoencoder.fit(X_train, X_train,
                            epochs=50,
                            batch_size=16,
                            shuffle=True,
                            validation_split = 0.1)
# validation_set ?
#autoencoder.summary()

# Summarize history for loss
plt.figure()
plt.plot(hist_auto.history['loss'])
plt.plot(hist_auto.history['val_loss'])
plt.title('Autoencoder model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# Predict on test set
encoded_data = encoder.predict(X_test)
encoded_data_prob = encoded_data[:,0]

predictions = np.where(encoded_data_prob > 0.5, 1, 0)

# Convert one-hot labels
rounded_labels = np.argmax(y_test, axis=1)

# Print accuracy
acc = accuracy_score(rounded_labels, predictions)
print('Overall accuracy of Neural Network model:', acc)

# Compute and visualize the confusion matrix
cm = confusion_matrix(rounded_labels, predictions)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", vmin = 0.5);
plt.title('Confusion Matrix')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()

# Vizualize the encoded part
encoder = Model(input_data, encoded)
encoded_data = encoder.predict(X_test)

encoded_data_df = pd.DataFrame(encoded_data)

plt.scatter(encoded_data_df.loc[:,0], encoded_data_df.loc[:,1], c = y.tolist(), s = 30)
encoded_data


##################### ALTERNATIVE ############################
# https://towardsdatascience.com/build-the-right-autoencoder-tune-and-optimize-using-pca-principles-part-i-1f01f821999b
# Fit Autoencoder
# nb_epoch = 100
# batch_size = 16 # Too much ??
# input_dim = X_train.shape[1] #num of predictor variables, 
# encoding_dim = 2
# learning_rate = 1e-3

# encoder = Dense(encoding_dim, activation="linear", input_shape=(input_dim,), use_bias = True) 
# decoder = Dense(input_dim, activation="linear", use_bias = True)

# autoencoder = Sequential()
# autoencoder.add(encoder)
# autoencoder.add(decoder)

# autoencoder.compile(metrics=['accuracy'],
#                     loss='mean_squared_error',
#                     optimizer='sgd')
# autoencoder.summary()

# autoencoder.fit(X_train_scaled, X_train_scaled,
#                 epochs=nb_epoch,
#                 batch_size=batch_size,
#                 shuffle=True,
#                 verbose=0)

# # Get the weights
# w_encoder = np.round(autoencoder.layers[0].get_weights()[0], 2).T
# w_decoder = np.round(autoencoder.layers[1].get_weights()[0], 2)  # W' in Figure 3.
# print('Encoder weights \n', w_encoder)
# print('Decoder weights \n', w_decoder)

# np.round(np.dot(w_encoder, w_encoder.T), 3)


























