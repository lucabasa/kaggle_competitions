import pandas as pd
import numpy as np

import tubesml as tml
from tubesml.base import BaseTransformer

from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

from src.sharpe import score_sharpe

import matplotlib.pyplot as plt


class TSCrossValidate(tml.CrossValidate):

    def _initialize_loop(self):
        """
        Prepares everything needed to loop over the folds.
        The estimator must be a pipeline, so we make it one if it isn't
        """
        if self.class_pos is None:
            self.oof = np.zeros((len(self.train), self.target.nunique()))
        else:
            self.oof = np.array([np.nan] * len(self.train))
        if self.df_test is not None:
            if self.class_pos is None:
                self.pred = np.zeros((len(self.df_test), self.target.nunique()))
            else:
                self.pred = np.zeros(len(self.df_test))
        else:
            self.pred = self.oof
        self.result_dict = {}

        self.feat_df = pd.DataFrame()
        self.iteration = []
        self.feat_pdp = pd.DataFrame()
        self.shap_values = np.ndarray(shape=(0, 1))

        if self.fit_params is None:
            self.fit_params = {}

        try:  # If estimator is not a pipeline, make a pipeline
            self.estimator.steps
        except AttributeError:
            self.estimator = Pipeline([("transf", BaseTransformer()), ("model", self.estimator)])


    def score(self):
        """
        Main method to loop over the folds, train and predict. It produces out of fold predictions
        and, if provided, an average prediction on the test set. It can also produce various insights
        on the model, like feature importance and pdp's.
        """
        folds_res = []
        for n_fold, (train_index, test_index) in enumerate(self.cv.split(self.train.values)):
            trn_data = self.train.iloc[train_index, :].reset_index(drop=True)
            val_data = self.train.iloc[test_index, :].reset_index(drop=True)

            og_val_data = val_data.copy()
            og_trn_data = trn_data.copy()

            trn_target, val_target = self._get_train_val_target(train_index, test_index)

            trn_data, val_data, test_data, model, transf_pipe = self._prepare_cv_iteration(
                trn_data, val_data, trn_target
            )
            
            if self.early_stopping:
                # Fit the model with early stopping
                if "sample_weight" in self.fit_params.keys():
                    sw = og_trn_data[self.fit_params["sample_weight"]]
                    new_params = {k: v for k, v in self.fit_params.items() if k != "sample_weight"}
                    model.fit(
                        trn_data, trn_target, eval_set=[(trn_data, trn_target), (val_data, val_target)], sample_weight=sw, **new_params
                    )
                else:
                    model.fit(
                        trn_data, trn_target, eval_set=[(trn_data, trn_target), (val_data, val_target)], **self.fit_params
                    )
                # store iteration used
                try:
                    self.iteration.append(model.best_iteration)
                except AttributeError:
                    self.iteration.append(model.best_iteration_)
            else:
                if "sample_weight" in self.fit_params.keys():
                    sw = og_trn_data[self.fit_params["sample_weight"]]
                    new_params = {k: v for k, v in self.fit_params.items() if k != "sample_weight"}
                    model.fit(trn_data, trn_target, sample_weight=sw, **new_params)
                else:
                    model.fit(trn_data, trn_target, **self.fit_params)

            if self.predict_proba:
                if self.class_pos is None:
                    predictions = model.predict_proba(val_data)[:, :]
                    self.oof[test_index] = predictions
                else:
                    predictions = model.predict_proba(val_data)[:, self.class_pos]
                    self.oof[test_index] = predictions
                if self.df_test is not None:
                    if self.class_pos is None:
                        self.pred += model.predict_proba(test_data)[:, :]
                    else:
                        self.pred += model.predict_proba(test_data)[:, self.class_pos]
            else:
                predictions = model.predict(val_data).ravel()
                self.oof[test_index] = predictions
                if self.df_test is not None:
                    self.pred += model.predict(test_data).ravel()

            if self.imp_coef:
                self._fold_imp(model, trn_data, n_fold)

            if self.pdp is not None:
                self._fold_pdp(model, transf_pipe, n_fold)

            if self.shap:
                self._fold_shap(model, trn_data)

            folds_res.append(fold_evaluation(test_set=og_val_data,
                                             target=val_target,
                                             predictions=predictions,
                                             fold_n=n_fold))

        self._summarize_results()

        self.result_dict["folds_eval"] = pd.concat(folds_res, ignore_index=True)

        if self.df_test is None:
            self.pred = None
            return self.oof, self.result_dict
        else:
            self._postprocess_prediction()
            return self.oof, self.pred, self.result_dict


def fold_evaluation(test_set, target, predictions, fold_n):
    df_eval = test_set.copy()
    df_eval["target"] = target
    df_eval["predictions"] = predictions
    df_eval["fold"] = fold_n

    min_date = df_eval["date_id"].min()
    df_eval["n_days"] = df_eval["date_id"] - min_date + 1

    df_eval["error"] = df_eval["target"] - df_eval["predictions"]
    df_eval["error_sqr"] = df_eval["error"]**2

    return df_eval


def summary_evaluation(df_eval, orig_df, factor=400):
    print(f"Mean Squared Error: {df_eval['error_sqr'].mean().round(5)}")
    print(f'R2: {r2_score(y_true=df_eval["target"], y_pred=df_eval["predictions"])}')
    print(f"MAE: {mean_absolute_error(y_true=df_eval['target'], y_pred=df_eval['predictions'])}")
    print(df_eval.groupby("fold")["error_sqr"].agg(["min", "mean", "max"]))

    df_eval["prediction"] = np.clip(df_eval["predictions"] * factor + 1, 0, 2)
    print(f"Sharpe: {score_sharpe(solution=orig_df[orig_df["date_id"] >= df_eval["date_id"].min()].reset_index(drop=True),
                                  submission=df_eval, row_id_column_name='')}")
    for n, group in df_eval.groupby("fold"):
        print(n, score_sharpe(solution=orig_df[(orig_df["date_id"] >= group["date_id"].min()) &
                                               (orig_df["date_id"] <= group["date_id"].max())].reset_index(drop=True).copy(),
                              submission=group.reset_index(), row_id_column_name=""))

    df_eval.groupby("n_days")[["error_sqr"]].quantile([.25, .50, .75]).unstack(1).plot()
    plt.show()

    df_eval["prediction"].plot()
    plt.show()

    tml.plot_regression_predictions(data=df_eval, true_label=df_eval["target"], pred_label=df_eval["predictions"])
