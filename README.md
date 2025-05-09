# DeepLearning24_25_NovaIMS
## Abstract

This project tackles the challenge of classifying rare biological species at the family level using deep learning models and the *BioCLIP* dataset. The dataset contains highly imbalanced and visually diverse images sourced from the Encyclopedia of Life, making this a complex multi-class classification task. We evaluated both custom convolutional neural networks and state-of-the-art pre-trained models, including *InceptionResNetV2*, *DenseNet121*, and *ConvNeXtBase*. Our pipeline involved architecture-specific preprocessing, careful train-validation-test splitting, and regularization strategies such as data augmentation and dropout. Among all approaches, the *ConvNeXtFull* model achieved the best performance, reaching a macro F1 score of 82% on a 90/10 split. While fine-tuning showed limited benefits and overfitting remained a concern, our results highlight the importance of appropriate preprocessing, balanced augmentation, and model selection in achieving strong generalization performance. This work demonstrates the potential of transfer learning in biodiversity research and opens opportunities for further optimization.

**Keywords:** Convolutional Neural Networks, Image Classification, Transfer Learning, Deep Learning, Data Augmentation, Keras, Regularization

## 🚀 Key Findings
Best Model: The ConvNeXtFull architecture achieved the highest performance with a macro F1 score of 82% on a 90/10 train-validation split.

Transfer Learning Wins: Pre-trained models (ConvNeXtBase, InceptionResNetV2, DenseNet121) significantly outperformed custom CNNs, saving training time and improving accuracy.

Data Imbalance: Class imbalance severely impacted model performance on rare species, especially in underrepresented families.

Augmentation Insights:

Beneficial for InceptionResNetV2 with moderate transformations.

Detrimental for ConvNeXtBase, where raw image input yielded better results.

Fine-Tuning Limitations: Attempts at fine-tuning often led to overfitting; frozen backbones provided more stable results.

Hyperparameter Tuning: Manual tuning outperformed random search, which often caused overfitting despite some strong configurations.

Data Volume Impact: Increasing training set size to 90% of data had the most substantial positive effect on generalization and validation performance.
