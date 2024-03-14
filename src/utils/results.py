import pandas as pd
import src.utils.plot as uplot
import numpy as np
from sklearn.preprocessing import LabelEncoder
import ast
import os
import plotly.express as px
import re
from src.utils.load import load_classifier
from sklearn.metrics import classification_report, f1_score


class ResultsManager():
    def __init__(self, config) -> None:
        self.config = config
        self.df_results = None
        self.le = None
        self.X_test = None
        pass

    def add_result_file(self, file_path, package):
        df = pd.read_csv(file_path, index_col=0)
        df['package'] = package
        if self.df_results is None:
            self.df_results = df
        else:
            self.df_results = pd.concat(
                [self.df_results, df]).reset_index(drop=True)
        self.df_results = self.df_results.drop_duplicates()

        return self

    def plot_f1_scores(self, filter_package=None, filter_model=None, figsize=(1200, 600), title=None):

        # filtre des doublons
        scores = self.df_results[[
            'model_path',
            'score_test',
            'package',
            'classifier',
            'vectorization']].reset_index(drop=True)
        index_to_keep = scores.groupby(['model_path'])['score_test'].idxmax()
        scores = scores.loc[index_to_keep].reset_index()

        # filtre des packages
        if filter_package is not None:
            scores = scores[
                (scores.package.isin(filter_package))
            ].reset_index(drop=True)
        if filter_model is not None:
            scores = scores[
                (scores.model_path.isin(filter_model))
            ].reset_index(drop=True)

        # ajout des colonnes pour le plot
        scores['serie_name'] = scores.apply(lambda row: row.classifier if pd.isna(
            row.vectorization) else row.classifier + ' - ' + row.vectorization, axis=1)
        scores['vectorizer'] = scores.apply(lambda row: row.classifier if pd.isna(
            row.vectorization) else row.vectorization, axis=1)

        # tri par score décroissant
        scores = scores[['serie_name', 'score_test',
                         'vectorizer']].reset_index()
        sorted_scores = scores.sort_values(by='score_test', ascending=False)

        # plot
        if title is None:
            title = 'Benchmark des f1 scores'
        uplot.plot_bench_results(
            sorted_scores,
            'serie_name',
            'score_test',
            'model',
            'f1 score',
            color_column='vectorizer',
            title=title,
            figsize=figsize
        )

        return self

    def plot_f1_scores_by_prdtype(self, filter_package=['text', 'img', 'bert']):
        scores = self.df_results[[
            'model_path', 'score_test', 'score_test_cat', 'package', 'classifier', 'vectorization']]
        scores.loc[:, 'vectorizer'] = scores.apply(lambda row: row.classifier if pd.isna(
            row.vectorization) else row.vectorization, axis=1)
        scores_index = scores.groupby(['model_path'])['score_test'].idxmax(
        )
        scores = scores.loc[scores_index].drop_duplicates()
        scores = scores[(scores.package.isin(filter_package))
                        & (scores.score_test_cat.notna())]
        scores_to_plot = None

        for index, row in scores.iterrows():
            score_to_plot = pd.DataFrame(ast.literal_eval(row.score_test_cat))
            score_to_plot.columns = ['score_test']
            score_to_plot['model'] = row.model_path
            score_to_plot['category'] = self.get_cat_labels()

            if scores_to_plot is None:
                scores_to_plot = score_to_plot
            else:
                scores_to_plot = pd.concat([scores_to_plot, score_to_plot])
        scores_to_plot = scores_to_plot.sort_values(
            by='score_test', ascending=False)

        fig = px.bar(
            scores_to_plot,
            x='category',
            y='score_test',
            color='model',
            barmode='group',
        )

        fig.update_traces(
            width=0.2,
        )
        # Update layout to remove legend and adjust xaxis title
        fig.update_layout(
            legend=None,
            xaxis_title='y_label',
            yaxis_title='x_label',
            # bargap=0.3,
            # bargroupgap=0.2,
            width=1200,
            height=600,
            title='title',
            # yaxis_range=[0.60, 1]
        )

        # Show the plot
        fig.show()
        return self

    def get_cat_labels(self):
        if self.le is None:
            self.le = LabelEncoder()
            self.le.classes_ = np.load(os.path.join(
                self.config.path_to_data, 'le_classes.npy'), allow_pickle=True)

        return self.le.classes_

    def get_num_classes(self):
        return len(self.get_cat_labels())

    def plot_classification_report(self, model_path, model_label=None):
        y_pred = self.get_y_pred(model_path)
        y_test = self.get_y_test(model_path)

        uplot.classification_results(
            y_test,
            y_pred,
            index=self.get_cat_labels(),
            model_label=model_label
        )
        return self

    def plot_confusion_matrix(self, model_path, model_label=None):
        y_pred = self.get_y_pred(model_path)
        y_test = self.get_y_test(model_path)

        uplot.plot_confusion_matrix(
            y_test,
            y_pred,
            index=self.get_cat_labels(),
            model_label=model_label
        )
        return self

    def plot_f1_scores_report(self, model_path, model_label=None):
        y_pred = self.get_y_pred(model_path)
        y_test = self.get_y_test(model_path)

        print(classification_report(y_test, y_pred,
              target_names=self.get_cat_labels()))

        return self

    def plot_classification_report_merged(self, model_paths):
        y_pred = []
        y_test = self.get_y_test(model_paths[0])
        for model_path in model_paths:
            y_pred.append(self.get_y_pred(model_path))

        y_merged = []
        for i, y in enumerate(y_test):
            merged_label = y_pred[0][i]
            for y_p in y_pred:
                if y_p[i] == y:
                    merged_label = y
                    break
            y_merged.append(merged_label)

        uplot.classification_results(
            y_test, y_merged, index=self.get_cat_labels())
        return self

    def get_y_pred(self, model_path):
        if pd.isna(self.df_results[self.df_results.model_path == model_path].pred_test.values[0]):
            clf = load_classifier(model_path)
            y_pred = clf.predict(self.get_X_test())
            return y_pred
        pred = self.df_results[self.df_results.model_path ==
                               model_path].pred_test.values[0]

        return ast.literal_eval(pred)

    def get_y_pred_probas(self, model_path):
        if pd.isna(self.df_results[self.df_results.model_path == model_path].probs_test.values[0]):
            clf = load_classifier(model_path)

            if hasattr(clf, 'predict_proba'):
                probas = clf.predict_proba(self.get_X_test())
            else:
                probas = np.zeros(
                    (len(self.get_X_test()), self.get_num_classes()))
                y_pred = self.get_y_pred(model_path)
                for i, y in enumerate(y_pred):
                    probas[i, y] = 1

            return probas
        return ast.literal_eval(self.df_results[self.df_results.model_path ==
                                                model_path].probs_test.values[0])

    def get_y_test(self, model_path):
        if pd.isna(self.df_results[self.df_results.model_path == model_path].y_test.values[0]):
            df = pd.read_csv(os.path.join(
                self.config.path_to_data, 'df_test_index.csv'))
            return df.prdtypeindex.values

        return ast.literal_eval(self.df_results[self.df_results.model_path == model_path].y_test.values[0])

    def get_X_test(self):
        if self.X_test is None:
            df = pd.read_csv(os.path.join(
                self.config.path_to_data, 'df_test_index.csv'))
            colnames = ['designation_translated', 'description_translated']
            df['tokens'] = df[colnames].apply(lambda row: ' '.join(
                s.lower() for s in row if isinstance(s, str)), axis=1)
            df['img_path'] = df.apply(lambda row:
                                      os.path.join(self.config.path_to_images, 'image_'
                                                   + str(row['imageid'])
                                                   + '_product_'
                                                   + str(row['productid'])
                                                   + '_resized'
                                                   + '.jpg'),
                                      axis=1)
            self.X_test = df[['tokens', 'img_path']]
        return self.X_test

    def get_model_paths(self):
        return self.df_results.model_path.unique()

    def get_model_label(self, model_path):
        label = self.df_results[self.df_results.model_path ==
                                model_path].classifier.values[0]
        if not pd.isna(self.df_results[self.df_results.model_path == model_path].vectorization.values[0]):
            label += '(' + self.df_results[self.df_results.model_path ==
                                           model_path].vectorization.values[0] + ')'

        return label

    def get_f1_score(self, model_path):
        return self.df_results[self.df_results.model_path == model_path].score_test.values[0]

    def voting_pred(self, basenames):
        probas = []
        weight_set = []

        for basename in basenames:
            probas.append(np.array(self.get_y_pred_probas(basename)))
            weight_set.append(self.get_f1_score(basename))

        probas_weighted = np.sum([probas[i] * weight_set[i]
                                  for i in range(len(probas))], axis=0)
        y_pred = np.argmax(probas_weighted, axis=1)
        return y_pred

    def voting_pred_cross_validate(self, basenames, n_folds=5, dataset_size=0.5):
        f_score_cv = []
        probas = []
        weight_set = []

        for basename in basenames:
            probas.append(np.array(self.get_y_pred_probas(basename)))
            weight_set.append(self.get_f1_score(basename))

        y_test = np.array(self.get_y_test(basenames[0]))
        test_idx_start = round((len(y_test) * (1-dataset_size)))
        test_size = len(y_test) - test_idx_start

        for k in range(n_folds):

            idx_start = test_idx_start + round((k-1)*(test_size/n_folds))
            idx_end = test_idx_start + round((k)*(test_size/n_folds) + 1)
            idx_fold = range(idx_start, idx_end)

            train_mask = np.ones(len(y_test), dtype=bool)
            train_mask[:test_idx_start] = False
            train_mask[idx_fold] = False

            test_mask = np.zeros(len(y_test), dtype=bool)
            test_mask[idx_fold] = True

            for basename in basenames:
                probas.append(np.array(self.get_y_pred_probas(basename)))
                y_pred = np.array(self.get_y_pred(basename))
                weight_set.append(
                    f1_score(y_test[train_mask], y_pred[train_mask], average='weighted'))

            probas_weighted = np.sum([probas[i] * weight_set[i]
                                      for i in range(len(probas))], axis=0)
            y_pred = np.argmax(probas_weighted, axis=1)
            f_score_cv.append(
                f1_score(y_test[test_mask], y_pred[test_mask], average='weighted'))

        print(f_score_cv)
        return np.mean(f_score_cv)
