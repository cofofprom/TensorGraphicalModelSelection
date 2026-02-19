from model import VectorGraphicalModel
from sklearn.covariance import graphical_lasso
from utils import matrix2Edges, evaluate
import argparse
import os
import json
import numpy as np
import pandas as pd
import uuid


def evaluateSingleModel(model, config, n_model, d):
    result = []

    for reg_param in np.logspace(1e-4, 1e-1, 100):
        for n_data in range(config['S_obs']):
            data = model.rvs(config['n_samples'])
            
            emp_cov = np.cov(data, rowvar=False)
            pred_precision = graphical_lasso(emp_cov, alpha=reg_param)[1]
    
            true_edges = matrix2Edges(model.precision)
            pred_edges = matrix2Edges(pred_precision)
        
            true_density = np.sum(true_edges) / len(true_edges)
            pred_density = np.sum(pred_edges) / len(pred_edges)
        
            result.append([true_density, pred_density, n_model, n_data, d, reg_param])

    return pd.DataFrame(result, columns=['true_density', 'pred_density', 'n_model', 'n_data', 'd', 'lambda'])

def evaluateSetOfModels(config):
    result = []
    for d in np.linspace(0.1, 0.9, 3):
        for n_model in range(config['S_sg']):
            model = VectorGraphicalModel(config['dim'], d)

            single_results = evaluateSingleModel(model, config, n_model, d)

            result.append(single_results)

    return pd.concat(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_dir', type=str, action='store')

    args = parser.parse_args()

    os.makedirs(args.experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(args.experiment_dir, 'data'), exist_ok=True)

    with open(os.path.join(args.experiment_dir, 'config.json'), 'r') as f:
        config = json.load(f)

    result_df = evaluateSetOfModels(config)

    fname = f'{uuid.uuid4()}.csv'
    result_df.to_csv(os.path.join(args.experiment_dir, 'data', fname), index=False)

if __name__ == '__main__':
    main()
