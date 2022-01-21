import os
import sys
import pandas as pd
from easydict import EasyDict as edict
import matplotlib.pyplot as plt

from configs import TrainConfig
from rich.console import Console
from rich.progress import Progress
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import ConvergenceWarning

from data.build_data import load_data
from ML_trainer.build_model import build_model
from utils_.utils import set_seed, get_scaler
from utils_.metrics import return_result
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

import warnings
warnings.filterwarnings(action='ignore')


def main():

    config = TrainConfig.parse_arguments()
    console = Console(color_system='256', force_terminal=True, width=160)
    console.log(config.__dict__)
    config.save()

    set_seed(config.random_state)

    # load data
    X, y = load_data(data_dir=config.data_dir,
                     target=config.target,
                     verbose=config.verbose)

    # train/test split samples
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=(1-config.train_ratio),
                                                        shuffle=False,
                                                        random_state=config.random_state)

    # scaling data
    scaler = get_scaler(config.scaler)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model_params = build_model(random_state=config.random_state)

    if config.model_names[0] == "all":
        model_list = list(model_params.keys())
    else:
        model_list = config.model_names

    # save results
    result_df = pd.DataFrame()


    # Training!
    with console.status(f"[bold green] Working on tasks...") as status:
        for model_name in model_list:
            model, model_param = model_params[model_name]['model'], model_params[model_name]['params']

            gcv = GridSearchCV(estimator=model, param_grid=model_param, n_jobs=3)
            gcv.fit(X_train, y_train)

            train_result = return_result(gcv.best_estimator_, X_train, y_train)
            test_result = return_result(gcv.best_estimator_, X_test, y_test)

            result = edict()
            result.random_state = config.random_state
            result.model_name = model_name

            for key in test_result.keys():
                result[f'train_{key}'] = train_result[key]
                result[f'test_{key}'] = test_result[key]

            result = pd.DataFrame.from_dict([result])
            result_df = pd.concat([result_df, result])

            # save result_df
            result_df.to_csv(os.path.join(config.checkpoint_dir, "result.csv"), index=False)

            # confusion matrix
            if config.confusion_matrix:
                label = sorted(y_test.unique().tolist())
                plot = plot_confusion_matrix(gcv.best_estimator_,  # 분류 모델
                                             X_test,
                                             y_test,  # 예측 데이터와 예측값의 정답(y_true)
                                             display_labels=label,  # 표에 표시할 labels
                                             cmap=plt.cm.Blues,  # 컬러맵(plt.cm.Reds, plt.cm.rainbow 등이 있음)
                                             normalize=None)
                plt.savefig(os.path.join(config.checkpoint_dir, f"{model_name}_confusion_matrix.png"), dpi=350)

            console.log(f"{model_name} model complete!")



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit()