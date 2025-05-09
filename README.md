# DeepLearning24_25_NovaIMS
## Abstract

This project tackles the challenge of classifying rare biological species at the family level using deep learning models and the *BioCLIP* dataset. The dataset contains highly imbalanced and visually diverse images sourced from the Encyclopedia of Life, making this a complex multi-class classification task. We evaluated both custom convolutional neural networks and state-of-the-art pre-trained models, including *InceptionResNetV2*, *DenseNet121*, and *ConvNeXtBase*. Our pipeline involved architecture-specific preprocessing, careful train-validation-test splitting, and regularization strategies such as data augmentation and dropout. Among all approaches, the *ConvNeXtFull* model achieved the best performance, reaching a macro F1 score of 82% on a 90/10 split. While fine-tuning showed limited benefits and overfitting remained a concern, our results highlight the importance of appropriate preprocessing, balanced augmentation, and model selection in achieving strong generalization performance. This work demonstrates the potential of transfer learning in biodiversity research and opens opportunities for further optimization.

**Keywords:** Convolutional Neural Networks, Image Classification, Transfer Learning, Deep Learning, Data Augmentation, Keras, Regularization



## ✅ Key Results

- **Best Model**: `ConvNeXtFull` variant  
- **Macro F1 Score**: **82%** on a 90/10 train-validation split  
- **Transfer learning** significantly outperformed custom CNNs  
- Fine-tuning led to **overfitting**; frozen backbones performed better  
- **Data volume** had the most positive impact on performance

## 🧰 Tools & Techniques

- **Frameworks**: TensorFlow, Keras, scikit-learn  
- **Models**: ConvNeXtBase, DenseNet121, InceptionResNetV2  
- **Strategies**:
  - Architecture-specific `preprocess_input`
  - Stratified train/val/test splits
  - Regularization: dropout, early stopping
  - Augmentation (selective use)
  - Hyperparameter tuning via Keras Tuner (Random Search)
