# Recommender System

This project is a clean and lightweight implementation of two popular recommendation system algorithms: Neural Collaborative Filtering (NCF) and Deep Factorization Machine (DFM). It is designed to be easy to understand and use for anyone interested in the field of recommendation systems.

## Getting Started

### Prerequisites

Make sure you have Python installed on your system. All required packages are listed in `requirements.txt`.

### Installation

Clone the repository to your local machine:
```bash
git clone https://github.com/ChenXuanting/RecommenderSystem.git
```
Navigate to the cloned repository, and install the required packages:
```bash
pip install -r requirements.txt
```
### Datasets

The project comes with a preprocessed version of the classic MovieLens 100K dataset in CSV format, located in the `dataset` folder.

Additionally, you can use Amazon review datasets by modifying the `ratings_name` variable in `main.py`. The available Amazon dataset names can be checked at [Amazon Review Datasets](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/).

## Usage

To train a model, run the following command:

python main.py <model_name>

Available model names are:
- `NCF`: For Neural Collaborative Filtering
- `DFM`: For Deep Factorization Machine

For example, to train the Neural Collaborative Filtering model, run:
```bash
python main.py NCF
```
### Hyperparameter Tuning

You can tune the hyperparameters of the models directly in the `main.py` file.

### Evaluation

The performance of the models is evaluated using two metrics:
- Hit Rate at K
- Normalized Discounted Cumulative Gain (NDCG)

We have implemented negative sampling and leave-one-out evaluation as proposed in the original papers. Related parameters, such as the number of negative samples, can be adjusted in the `main.py` file.

## Contributing

Contributions to improve the project are welcome. Feel free to fork the repository and submit pull requests.

## License

This project is open-source and available under the MIT License.
