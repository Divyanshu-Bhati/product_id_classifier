Data preparation:
 - Source: https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews/data
 - Created train-test-eval splits using ProductID and UserID columns.
 - Detailed thought process, EDA and preprocessing pipeline in notebooks/prepare_dataset.ipynb

Future work:
 - Currently, the model assumes the entire input must be classified. Real world data is often not that clear, so a segmentation approach which creates inputs from a stream of text could be interesting.
 - In custom VAE model, current approach sets `x_flat = x_emb.view(x.size(0), -1)` which treats the sequence as one big vector and effectively ignores positional information (for example, char at position 5 may be related to position 10). It could be interesting to consider a positional encoding approach.