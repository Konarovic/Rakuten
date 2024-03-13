import pandas as pd
import src.utils.plot as uplot
import numpy as np
from sklearn.preprocessing import LabelEncoder
import ast
import os
import plotly.express as px
import re
from src.utils.load import load_classifier


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
            self.df_results = pd.concat([self.df_results, df])
        self.df_results = self.df_results.drop_duplicates()

        return self

    def plot_f1_scores(self, filter_package=['text', 'img', 'bert']):
        scores = self.df_results[[
            'model_path', 'score_test', 'package', 'classifier', 'vectorization']]
        scores.loc[:, 'vectorizer'] = scores.apply(lambda row: row.classifier if pd.isna(
            row.vectorization) else row.vectorization, axis=1)
        scores = scores.groupby(['model_path', 'package', 'classifier', 'vectorizer']).max(
            'score_test').reset_index()
        scores = scores[scores.package.isin(filter_package)]
        sorted_scores = scores.sort_values(by='score_test', ascending=False)
        uplot.plot_bench_results(
            sorted_scores,
            'model_path',
            'score_test',
            'model',
            'f1 score',
            color_column='vectorizer',
            title='Benchmark des f1 scores'
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

    def plot_classification_report(self, model_path):
        y_pred = self.get_y_pred(model_path)
        y_test = self.get_y_test(model_path)

        uplot.classification_results(
            y_test, y_pred, index=self.get_cat_labels())
        return self

    def get_y_pred(self, model_path):
        if pd.isna(self.df_results[self.df_results.model_path == model_path].pred_test.values[0]):
            clf = load_classifier(model_path)
            y_pred = clf.predict(self.get_X_test())
            return y_pred
        pred = self.df_results[self.df_results.model_path ==
                               model_path].pred_test.values[0]

        return ast.literal_eval(pred)

    def get_y_test(self, model_path):
        if pd.isna(self.df_results[self.df_results.model_path == model_path].y_test.values[0]):
            df = pd.read_csv(os.path.join(
                self.config.path_to_data, 'df_test_index.csv'))
            return df.prdtypeindex.values

        return ast.literal_eval(self.df_results[self.df_results.model_path == model_path].y_test.values[0])

    def get_X_test(self):
        print(os.path.join(
            self.config.path_to_data, 'df_test_index.csv'))
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
