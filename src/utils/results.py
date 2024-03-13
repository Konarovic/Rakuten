import pandas as pd
import src.utils.plot as uplot
import numpy as np
from sklearn.preprocessing import LabelEncoder
import config
import ast
import os
import plotly.express as px


class ResultsManager():
    def __init__(self) -> None:
        self.df_results = None
        self.le = None
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
                config.path_to_data, 'le_classes.npy'), allow_pickle=True)

        return self.le.classes_


# def plot_bench_results_cat(self, data, x_column, y_column, x_label, y_label, color_column=None, title=None):

#     custom_categories_order = data[x_column].unique().tolist()

#     for i, cat in enumerate(custom_categories_order):
#         custom_categories_order[i] = self.le.inverse_transform(
#             ast.literal_eval(cat

#     fig = px.bar(
#         data,
#         y=x_column,
#         x=y_column,
#         color=color_column,
#         color_discrete_sequence=px.colors.qualitative.Plotly,
#         category_orders={x_column: custom_categories_order},
#     )

#     fig.update_traces(
#         width=0.8,

#     )
#     # Update layout to remove legend and adjust xaxis title
#     fig.update_layout(
#         legend=None,
#         xaxis_title=y_label,
#         yaxis_title=x_label,
#         bargap=0.3,
#         bargroupgap=0.2,
#         barmode='stack',
#         width=1200,
#         height=600,
#         title=title

#     )

#     # Show the plot
#     fig.show()
#     return fig
