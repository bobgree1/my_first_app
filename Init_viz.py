#
#
# import base64
# from decimal import Decimal
# from xml.etree import ElementTree
#
# import dash  # (version 1.12.0) pip install dash
# import dash_bootstrap_components as dbc
# import dash_core_components as dcc
# import dash_html_components as html
# import joblib
# import numpy as np
# import pandas as pd
# import plotly.express as px  # (version 4.7.0)
# import plotly.graph_objects as go
# import pydot
# import toad
# from dash.dependencies import Input, Output
# from dash_table import DataTable
# from sklearn import tree
# from sklearn.linear_model import LogisticRegression as LR
# from sklearn.model_selection import train_test_split
#
# app = dash.Dash(__name__)
#
# # ------------------------------------------------------------------------------
# # Import and clean data (importing csv into pandas)
#
# data = pd.read_csv('all_data.csv')
# index_no = data.columns.get_loc('label')
# cols = data.columns.tolist()
# cols = cols[0:1] + cols[289:290] + cols[1:289] + cols[290:]
# data = data[cols]
# print('Shape', data.shape)
# data.head(10)
# lb = pd.read_csv('Comp_name.csv', index_col=0)
# lb.columns = ['exchange', 'code', 'company', 'industry', 'advantage', 'risk']
# for i in range(lb.shape[0]):
#     ind = data[data.code == lb.code[i]].index
#     data.loc[ind, 'risk'] = lb.risk[i]
# data = data[data.risk <= 11]
# data
# X = pd.concat([data.code, data.iloc[:, 2:]], axis=1)
# y = data.iloc[:, 1:2]
# X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.3)
# train = pd.concat([X_tr, y_tr], axis=1)
# train.index = np.arange(len(train))
# test = pd.concat([X_ts, y_ts], axis=1)
# test.index = np.arange(len(test))
#
# c = toad.transform.Combiner()
# ex_lis = ['code', 'label']
# train_selected = train.copy()
# # dropped = toad.selection.select(train,target = 'risk', empty=0.8,
# #  iv=0.02,corr=0.7,return_drop=True,exclude='code')
# c.fit(train_selected, y='risk', method='chi', min_samples=0.03, exclude=ex_lis)
#
# nncol = ['acs_roe_2016',
#          'fcf_2019',
#          'acs_np_inc_2019',
#          'roe_bin_2019',
#          'gv_inc_2019',
#          'ibdr_2019',
#          'eps_2019',
#          'alr_inc_2018',
#          'npr_2016',
#          'gpr_inc_2018',
#          'toar_2019',
#          'gpr_2019',
#          'ncf_rp_bin']
# # datasets=c.transform(train_selected[ncol+['risk']]).copy()
# datasets = c.transform(train_selected[nncol + ['risk']].copy())
# x = datasets.drop('risk', axis=1).copy()
# y = datasets['risk'].copy()
#
# clf = tree.DecisionTreeClassifier(max_depth=7, random_state=0, splitter='best',
#                                   min_samples_leaf=20, criterion='entropy',
#                                   min_samples_split=10)
# clf.fit(x, y)
#
# # these columns are selcted by toad.selection.stepwise
# ncol = ['lvrg_inc_2018', 'gpr_inc_2018', 'eps_2019', 'gpr_2019', 'toar_2017', 'ncf_rp_bin']
# c2 = toad.transform.Combiner()
# ex_lis = ['code', 'risk']
# train_selected_lr = train_selected[ncol + ex_lis + ['label']]
# c2.fit(train_selected_lr, y='label', method='chi', min_samples=0.03, exclude=ex_lis)
# # these prarameters require further tuning when applied on different dataset
# adj_bin = {'gpr_2019': [0.1056, 0.2061, 0.3064, 0.4261],
#            'gpr_inc_2018': [-0.0829, -0.041, -0.02, -0.0028],
#            'lvrg_inc_2018': [-0.1584, -0.0018, 0.4661],
#            'toar_2017': [2.4592935284359823, 3.730076787579105, 5.654562771189577, 51.22],
#            'eps_2019': [0.0131, 0.1, .267, .44],
#            'ncf_rp_bin': [-2, -1, 0, 2],
#            }
# c2.set_rules(adj_bin)
#
# transer = toad.transform.WOETransformer()
# train_woe = transer.fit_transform(c2.transform(train_selected_lr),
#                                   train_selected_lr['label'], exclude=['code', 'risk', 'label'])
# test_woe = transer.transform(c2.transform(test))
#
# final_train = train_woe[ncol + ['label']]
# final_test = test_woe[ncol + ['label']]
# final_train[ncol], final_test[ncol] = -final_train[ncol], -final_test[ncol]
#
# tree_pred = pd.Series(clf.predict(c.transform(test[x.columns])))
# lr = LR(solver='liblinear').fit(train[ncol], final_train['label'])
# y_pred = lr.predict_proba(final_test[ncol])
# p_z = y_pred[:, 0]
# p_f = y_pred[:, 1]
# B = 20 / np.log(2)
# A = 650 + B * lr.intercept_
# score_lis = A + B * np.log(p_z / p_f) / np.log(2)
# cdf = pd.concat([tree_pred, pd.Series(score_lis),
#                  pd.Series(p_f), test.risk, test.code, test.industry], axis=1)
# cdf.columns = ['Tree_pred', 'Score', 'lr_pred', 'risk', 'code', 'industry']
# cdf['label'] = test.label.copy()
# gx, gy = cdf[cdf.label == 0].label, cdf[cdf.label == 0].Score
# bx, by = cdf[cdf.label == 1].label, cdf[cdf.label == 1].Score
#
# sc = pd.read_csv('ScoreData.csv')
# sc.columns = sc.columns.tolist()[:3] + ['score ']
# sc.fillna("")
#
#
# def svg_to_fig(svg_bytes, title=None, plot_bgcolor="white", x_lock=False, y_lock=True):
#     svg_enc = base64.b64encode(svg_bytes)
#     svg = f"data:image/svg+xml;base64, {svg_enc.decode()}"
#
#     # Get the width and height
#     xml_tree = ElementTree.fromstring(svg_bytes.decode())
#     img_width = int(xml_tree.attrib["width"].strip("pt"))
#     img_height = int(xml_tree.attrib["height"].strip("pt"))
#
#     fig = go.Figure()
#     # Add invisible scatter trace.
#     # This trace is added to help the autoresize logic work.
#     fig.add_trace(
#         go.Scatter(
#             x=[0, img_width],
#             y=[img_height, 0],
#             mode="markers",
#             marker_opacity=0,
#             hoverinfo="none",
#         )
#     )
#     fig.add_layout_image(
#         dict(
#             source=svg,
#             x=0,
#             y=0,
#             xref="x",
#             yref="y",
#             sizex=img_width,
#             sizey=img_height,
#             opacity=1,
#             layer="below",
#         )
#     )
#
#     # Adapt axes to the right width and height, lock aspect ratio
#     fig.update_xaxes(showgrid=False, visible=False, range=[0, img_width])
#     fig.update_yaxes(showgrid=False, visible=False, range=[img_height, 0])
#
#     if x_lock is True:
#         fig.update_xaxes(constrain="domain")
#     if y_lock is True:
#         fig.update_yaxes(scaleanchor="x", scaleratio=1)
#
#     fig.update_layout(plot_bgcolor=plot_bgcolor, margin=dict(r=5, l=5, b=5))
#
#     if title:
#         fig.update_layout(title=title)
#
#     return fig
#
#
# MODELS = ['Deep', 'Shallow', 'Entropy', 'Random']
# output_card = dbc.Card(
#     [
#         dbc.CardHeader("Real-time Prediction"),
#         dbc.CardBody(html.H5(id="predicted-grade", style={"text-align": "center"})),
#     ]
# )
# model_selection = dbc.InputGroup(
#     [
#         dbc.InputGroupAddon("Select Model", addon_type="prepend"),
#         dbc.Select(
#             id="model-selection",
#             options=[
#                 {
#                     "label": m.replace("-", " ").capitalize(),
#                     "value": "assets/" + m + ".joblib",
#                 }
#                 for m in MODELS
#             ],
#             value="assets/" + MODELS[0] + ".joblib",
#         ),
#     ]
# )
#
# sample_controls = [
#     dbc.Col(
#         dbc.ButtonGroup(
#             [
#                 dbc.Button(
#                     "Prev. Sample",
#                     id="prev-sample",
#                     color="info",
#                     n_clicks=0, outline=True,
#                 ),
#                 dbc.Button(
#                     "Next Sample", id="next-sample", n_clicks=0, color="info",
#                 ),
#             ],
#             style={"width": "100%"}, className='mr-1'
#         )
#     ),
# ]
# #         A
# #        A A
# #       A A A
# # # #  A     A
#
# app.layout = html.Div([
#         html.H1("Risk & Credit Evaluation with LR and Decision Tree"),
#         html.Hr(),
#         # dbc.Row(controls, style={"padding": "20px 0px"}),
#         dbc.Row(
#             [
#                 dbc.Col(
#                     children=[
#                         dbc.CardDeck([output_card]),
#                         DataTable(
#                             id="table-sample",
#                             style_table={
#                                 "height": "400px",
#                                 "padding": "20px",
#                             },
#                             style_cell_conditional=[
#                                 {'if': {'column_id': 'Feature'},
#                                  'width': '20%'},
#                             ]
#                         ),
#
#                     ],
#                     md=2),
#
#                 dbc.Col(
#                     [dbc.Col(dbc.Row(sample_controls), md=7),
#                      html.Br(),
#                      DataTable(
#                          columns=[{'name': i, 'id': i}
#                                   for i in sc.columns],
#                          data=sc.to_dict('records'),
#                          style_data={  # overflow cells' content into multiple lines
#                              'whiteSpace': 'normal',
#                              "height": "500px",
#                              'height': 'auto'
#                          },
#                          style_table={
#                              "overflowY": "auto",
#                          },
#                          tooltip_data=[
#                              {
#                                  column: {'value': str(value), 'type': 'markdown'}
#                                  for column, value in row.items()
#                              } for row in sc.to_dict('records')
#                          ],
#                          style_cell={
#                              'overflow': 'hidden',
#                              'textOverflow': 'ellipsis',
#                              'maxWidth': 0,
#                          },
#                      ), ],
#
#                     md=4),
#                 dbc.Col(children=[
#                     dbc.Col(model_selection, md=6),
#                     dcc.Graph(id="graph-tree", style={"height": "600px"}),
#                 ], md=6, ),
#             ]
#         ),
#         dbc.Row([dbc.Col(dcc.Graph(id='2D', style={'height': '500px'}), md=4),
#                  dbc.Col(dcc.Graph(id='score_summary', style={'height': '500px'}), md=2),
#                  dbc.Col(dcc.Graph(id="3D", style={'height': '500px'}), md=6)]),
#
#     ],
#     style={"margin": "auto"},
# )
#
# tl = list(np.arange(11) + 1) + ['12']
# lis = [str(i) for i in tl]
# feature_names = x.columns
# class_names = lis
#
#
# @app.callback(Output("graph-tree", "figure"), [Input("model-selection", "value")])
# def visualize_tree(path):
#     model = joblib.load(open(path, "rb"))
#     dot_data = tree.export_graphviz(
#         model,
#         out_file=None,
#         filled=True,
#         rounded=True,
#         feature_names=feature_names,
#         class_names=class_names,
#         proportion=True,
#         rotate=True,
#         precision=2,
#     )
#
#     pydot_graph = pydot.graph_from_dot_data(dot_data)[0]
#     svg_bytes = pydot_graph.create_svg()
#     fig = svg_to_fig(svg_bytes, )  # title=""
#
#     return fig
#
#
# @app.callback(
#     [
#         Output("table-sample", "data"),
#         Output("table-sample", "columns"),
#         Output("predicted-grade", "children"),
#     ],
#     [
#         Input("prev-sample", "n_clicks"),
#         Input("next-sample", "n_clicks"),
#         Input("model-selection", "value"),
#     ],
# )
# def generate_table(prev_clicks, next_clicks, model_path):
#     dff = data.copy()
#
#     # Build the sample table
#     i = max(0, next_clicks - prev_clicks)
#     table_df = dff[nncol].loc[i:i].T.reset_index()
#     table_df.columns = ["Feature", "Cur_Val"]
#
#     # Load model and make prediction
#     model = joblib.load(open(model_path, "rb"))
#     model_input = dff[nncol].loc[i:i].copy()
#     pred = model.predict(c.transform(model_input))[0]
#
#     model_input = dff[ncol].loc[i:i].copy()
#     lr = joblib.load(open('assets/lr.joblib', "rb"))
#     # negative value problematic toad
#     arr = lr.predict_proba(-transer.transform(c2.transform(model_input)))
#     B = 20 / np.log(2)
#     A = 650 + B * lr.intercept_
#     score = A + B * np.log(arr[0, 0] / arr[0, 1]) / np.log(2)
#     pred_lr = np.round(score[0]).astype(int)
#
#     table_df = table_df.round(2)
#     table_df.iloc[table_df[table_df.Feature == 'fcf_2019'].index, 1] = '%.2E' % Decimal(
#         table_df.iloc[table_df[table_df.Feature == 'fcf_2019'].index, 1].values[0])
#     columns = [{"name": i, "id": i} for i in table_df.columns]
#
#     return table_df.to_dict("records"), columns, f"Grade = {pred}" + '    ' + f"Score = {pred_lr}"
#
#
# @app.callback(
#     Output('2D', 'figure'), [Input("model-selection", "value")]
# )
# def visualize_2dgraph(hi):
#     fig = px.scatter(cdf, x="Tree_pred", y="Score", color="risk", symbol='industry',
#                      hover_data=['code', 'Score'],
#                      color_continuous_scale='blues')
#     fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
#     fig.update_layout({'legend_orientation': 'h'})
#     fig.update_traces(showscale=False, selector=dict(type='isosurface'))
#     return fig
#
#
# @app.callback(
#     Output('score_summary', 'figure'), [Input("model-selection", "value")]
# )
# def vis_summary(hi):
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=gx, y=gy,
#         mode='markers',
#         marker={'size': 3.5},
#         marker_color='rgba(0,128,128,0.8)', name='Normal',
#         hovertext=['code=' + str(cdf['code'][i]) for i in range(cdf.shape[0])]
#     ))
#
#     fig.add_trace(go.Scatter(
#         x=bx, y=by,
#         mode='markers',
#         marker_color='rgba(128, 0, 0, .6)',
#         marker={'size': 2}, name='Risky',
#         hovertext=['code=' + str(cdf['code'][i]) for i in range(cdf.shape[0])]
#     ))
#     fig.update_layout({'legend_orientation': 'h'}, width=400, xaxis_range=[-0.25, 1.25], )
#
#     fig.update_layout(
#         xaxis_title="Risky (Binary)",
#         yaxis_title="Score",
#         font=dict(
#             family="Courier New, monospace",
#             size=14))
#
#     fig.update_layout(
#         title={'text': "Scorecard vs Label", 'y': 1, 'x': 0.5,
#                'xanchor': 'center',
#                'yanchor': 'top'}
#     )
#     fig.update_layout(margin=dict(l=0, r=0, b=0, t=20))
#     return fig
#
#
# @app.callback(
#     Output('3D', 'figure'), [Input("model-selection", "value")]
# )
# def visualize_graph(hi):
#     fig = px.scatter_3d(cdf, x="Tree_pred", y="Score", z="risk", symbol='industry',
#                         hover_data=['code'], color='risk',
#                         color_continuous_scale='greens')
#     fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
#     fig.update_layout({'legend_orientation': 'h'})
#     fig.update_traces(showscale=False, selector=dict(type='isosurface'))
#     return fig
#
#
# if __name__ == '__main__':
#     app.run_server()
# # app.run_server(mode='inline')


import pandas as pd
import base64, xml
from xml.etree import ElementTree
import plotly.express as px  # (version 4.7.0)
import plotly.graph_objects as go

import dash  # (version 1.12.0) pip install dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import glob, flask
import numpy as np
import toad
import math
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split
from decimal import Decimal

from sklearn import tree

import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash_table import DataTable
import plotly.graph_objects as go
import pydot

app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])
server=app.server
# ------------------------------------------------------------------------------
# Import and clean data (importing csv into pandas)

data = pd.read_csv('all_data.csv')
index_no = data.columns.get_loc('label')
cols = data.columns.tolist()
cols = cols[0:1] + cols[289:290] + cols[1:289] + cols[290:]
data = data[cols]
print('Shape', data.shape)
data.head(10)
lb = pd.read_csv('Comp_name.csv', index_col=0)
lb.columns = ['exchange', 'code', 'company', 'industry', 'advantage', 'risk']
for i in range(lb.shape[0]):
    ind = data[data.code == lb.code[i]].index
    data.loc[ind, 'risk'] = lb.risk[i]
data = data[data.risk <= 11]
data
X = pd.concat([data.code, data.iloc[:, 2:]], axis=1)
y = data.iloc[:, 1:2]
X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.3)
train = pd.concat([X_tr, y_tr], axis=1)
train.index = np.arange(len(train))
test = pd.concat([X_ts, y_ts], axis=1)
test.index = np.arange(len(test))

c = toad.transform.Combiner()
ex_lis = ['code', 'label']
train_selected = train.copy()
# dropped = toad.selection.select(train,target = 'risk', empty=0.8,
#  iv=0.02,corr=0.7,return_drop=True,exclude='code')
c.fit(train_selected, y='risk', method='chi', min_samples=0.03, exclude=ex_lis)

nncol = ['acs_roe_2016',
         'fcf_2019',
         'acs_np_inc_2019',
         'roe_bin_2019',
         'gv_inc_2019',
         'ibdr_2019',
         'eps_2019',
         'alr_inc_2018',
         'npr_2016',
         'gpr_inc_2018',
         'toar_2019',
         'gpr_2019',
         'ncf_rp_bin']
# datasets=c.transform(train_selected[ncol+['risk']]).copy()
datasets = c.transform(train_selected[nncol + ['risk']].copy())
x = datasets.drop('risk', axis=1).copy()
y = datasets['risk'].copy()

clf = tree.DecisionTreeClassifier(max_depth=7, random_state=0, splitter='best',
                                  min_samples_leaf=20, criterion='entropy',
                                  min_samples_split=10)
clf.fit(x, y)

# these columns are selcted by toad.selection.stepwise
ncol = ['lvrg_inc_2018', 'gpr_inc_2018', 'eps_2019', 'gpr_2019', 'toar_2017', 'ncf_rp_bin']
c2 = toad.transform.Combiner()
ex_lis = ['code', 'risk']
train_selected_lr = train_selected[ncol + ex_lis + ['label']]
c2.fit(train_selected_lr, y='label', method='chi', min_samples=0.03, exclude=ex_lis)
# these prarameters require further tuning when applied on different dataset
adj_bin = {'gpr_2019': [0.1056, 0.2061, 0.3064, 0.4261],
           'gpr_inc_2018': [-0.0829, -0.041, -0.02, -0.0028],
           'lvrg_inc_2018': [-0.1584, -0.0018, 0.4661],
           'toar_2017': [2.4592935284359823, 3.730076787579105, 5.654562771189577, 51.22],
           'eps_2019': [0.0131, 0.1, .267, .44],
           'ncf_rp_bin': [-2, -1, 0, 2],
           }
c2.set_rules(adj_bin)

transer = toad.transform.WOETransformer()
train_woe = transer.fit_transform(c2.transform(train_selected_lr),
                                  train_selected_lr['label'], exclude=['code', 'risk', 'label'])
test_woe = transer.transform(c2.transform(test))

final_train = train_woe[ncol + ['label']]
final_test = test_woe[ncol + ['label']]
final_train[ncol], final_test[ncol] = -final_train[ncol], -final_test[ncol]

tree_pred = pd.Series(clf.predict(c.transform(test[x.columns])))
lr = LR(solver='liblinear').fit(train[ncol], final_train['label'])
y_pred = lr.predict_proba(final_test[ncol])
p_z = y_pred[:, 0]
p_f = y_pred[:, 1]
B = 20 / np.log(2)
A = 650 + B * lr.intercept_
score_lis = A + B * np.log(p_z / p_f) / np.log(2)
cdf = pd.concat([tree_pred, pd.Series(score_lis),
                 pd.Series(p_f), test.risk, test.code, test.industry], axis=1)
cdf.columns = ['Tree_pred', 'Score', 'lr_pred', 'risk', 'code', 'industry']
cdf['label'] = test.label.copy()
gx, gy = cdf[cdf.label == 0].label, cdf[cdf.label == 0].Score
bx, by = cdf[cdf.label == 1].label, cdf[cdf.label == 1].Score

sc = pd.read_csv('ScoreData.csv')
sc.columns = sc.columns.tolist()[:3] + ['score ']
sc.fillna("")


def svg_to_fig(svg_bytes, title=None, plot_bgcolor="white", x_lock=False, y_lock=True):
    svg_enc = base64.b64encode(svg_bytes)
    svg = f"data:image/svg+xml;base64, {svg_enc.decode()}"

    # Get the width and height
    xml_tree = ElementTree.fromstring(svg_bytes.decode())
    img_width = int(xml_tree.attrib["width"].strip("pt"))
    img_height = int(xml_tree.attrib["height"].strip("pt"))

    fig = go.Figure()
    # Add invisible scatter trace.
    # This trace is added to help the autoresize logic work.
    fig.add_trace(
        go.Scatter(
            x=[0, img_width],
            y=[img_height, 0],
            mode="markers",
            marker_opacity=0,
            hoverinfo="none",
        )
    )
    fig.add_layout_image(
        dict(
            source=svg,
            x=0,
            y=0,
            xref="x",
            yref="y",
            sizex=img_width,
            sizey=img_height,
            opacity=1,
            layer="below",
        )
    )

    # Adapt axes to the right width and height, lock aspect ratio
    fig.update_xaxes(showgrid=False, visible=False, range=[0, img_width])
    fig.update_yaxes(showgrid=False, visible=False, range=[img_height, 0])

    if x_lock is True:
        fig.update_xaxes(constrain="domain")
    if y_lock is True:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

    fig.update_layout(plot_bgcolor=plot_bgcolor, margin=dict(r=5, l=5, b=5))

    if title:
        fig.update_layout(title=title)

    return fig

MODELS = ['Deep', 'Shallow', 'Entropy', 'Random']
output_card = dbc.Card(
    [
        dbc.CardHeader("Real-time Prediction"),
        dbc.CardBody(html.H5(id="predicted-grade", style={"text-align": "center"})),
    ]
)
model_selection = dbc.InputGroup(
    [
        dbc.InputGroupAddon("Select Model", addon_type="prepend"),
        dbc.Select(
            id="model-selection",
            options=[
                {
                    "label": m.replace("-", " ").capitalize(),
                    "value": "assets/" + m + ".joblib",
                }
                for m in MODELS
            ],
            value="assets/" + MODELS[0] + ".joblib",
        ),
    ]
)

sample_controls = [
    dbc.Col(
        dbc.ButtonGroup(
            [
                dbc.Button(
                    "Prev. Sample",
                    id="prev-sample",
                    color="info",
                    n_clicks=0, outline=True,
                ),
                dbc.Button(
                    "Next Sample", id="next-sample", n_clicks=0, color="info",
                ),
            ],
            style={"width": "100%"}, className='mr-1'
        )
    ),
]
#         A
#        A A
#       A A A
# # #  A     A

app.layout = dbc.Container(
    fluid=True,
    children=[
        html.H1("Risk & Credit Evaluation with LR and Decision Tree"),
        html.Hr(),
        # dbc.Row(controls, style={"padding": "20px 0px"}),
        dbc.Row(
            [
                dbc.Col(
                    children=[
                        dbc.CardDeck([output_card]),
                        DataTable(
                            id="table-sample",
                            style_table={
                                "height": "400px",
                                "padding": "20px",
                            },
                            style_cell_conditional=[
                                {'if': {'column_id': 'Feature'},
                                 'width': '20%'},
                            ]
                        ),

                    ],
                    md=2),

                dbc.Col(
                    [dbc.Col(dbc.Row(sample_controls), md=7),
                     html.Br(),
                     DataTable(
                         columns=[{'name': i, 'id': i}
                                  for i in sc.columns],
                         data=sc.to_dict('records'),
                         style_data={  # overflow cells' content into multiple lines
                             'whiteSpace': 'normal',
                             "height": "500px",
                             'height': 'auto'
                         },
                         style_table={
                             "overflowY": "auto",
                         },
                         tooltip_data=[
                             {
                                 column: {'value': str(value), 'type': 'markdown'}
                                 for column, value in row.items()
                             } for row in sc.to_dict('records')
                         ],
                         style_cell={
                             'overflow': 'hidden',
                             'textOverflow': 'ellipsis',
                             'maxWidth': 0,
                         },
                     ), ],

                    md=4),
                dbc.Col(children=[
                    dbc.Col(model_selection, md=6),
                    dcc.Graph(id="graph-tree", style={"height": "600px"}),
                ], md=6, ),
            ]
        ),
        dbc.Row([dbc.Col(dcc.Graph(id='2D', style={'height': '500px'}), md=4),
                 dbc.Col(dcc.Graph(id='score_summary', style={'height': '500px'}), md=2),
                 dbc.Col(dcc.Graph(id="3D", style={'height': '500px'}), md=6)]),

    ],
    style={"margin": "auto"},
)

tl = list(np.arange(11) + 1) + ['12']
lis = [str(i) for i in tl]
feature_names = x.columns
class_names = lis


@app.callback(Output("graph-tree", "figure"), [Input("model-selection", "value")])
def visualize_tree(path):
    model = joblib.load(open(path, "rb"))
    dot_data = tree.export_graphviz(
        model,
        out_file=None,
        filled=True,
        rounded=True,
        feature_names=feature_names,
        class_names=class_names,
        proportion=True,
        rotate=True,
        precision=2,
    )

    pydot_graph = pydot.graph_from_dot_data(dot_data)[0]
    svg_bytes = pydot_graph.create_svg()
    fig = svg_to_fig(svg_bytes, )  # title=""

    return fig


@app.callback(
    [
        Output("table-sample", "data"),
        Output("table-sample", "columns"),
        Output("predicted-grade", "children"),
    ],
    [
        Input("prev-sample", "n_clicks"),
        Input("next-sample", "n_clicks"),
        Input("model-selection", "value"),
    ],
)
def generate_table(prev_clicks, next_clicks, model_path):
    dff = data.copy()

    # Build the sample table
    i = max(0, next_clicks - prev_clicks)
    table_df = dff[nncol].loc[i:i].T.reset_index()
    table_df.columns = ["Feature", "Cur_Val"]

    # Load model and make prediction
    model = joblib.load(open(model_path, "rb"))
    model_input = dff[nncol].loc[i:i].copy()
    pred = model.predict(c.transform(model_input))[0]

    model_input = dff[ncol].loc[i:i].copy()
    lr = joblib.load(open('assets/lr.joblib', "rb"))
    # negative value problematic toad
    arr = lr.predict_proba(-transer.transform(c2.transform(model_input)))
    B = 20 / np.log(2)
    A = 650 + B * lr.intercept_
    score = A + B * np.log(arr[0, 0] / arr[0, 1]) / np.log(2)
    pred_lr = np.round(score[0]).astype(int)

    table_df = table_df.round(2)
    table_df.iloc[table_df[table_df.Feature == 'fcf_2019'].index, 1] = '%.2E' % Decimal(
        table_df.iloc[table_df[table_df.Feature == 'fcf_2019'].index, 1].values[0])
    columns = [{"name": i, "id": i} for i in table_df.columns]

    return table_df.to_dict("records"), columns, f"Grade = {pred}" + '    ' + f"Score = {pred_lr}"


@app.callback(
    Output('2D', 'figure'), [Input("model-selection", "value")]
)
def visualize_2dgraph(hi):
    fig = px.scatter(cdf, x="Tree_pred", y="Score", color="risk", symbol='industry',
                     hover_data=['code', 'Score'],
                     color_continuous_scale='blues')
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.update_layout({'legend_orientation': 'h'})
    fig.update_traces(showscale=False, selector=dict(type='isosurface'))
    return fig


@app.callback(
    Output('score_summary', 'figure'), [Input("model-selection", "value")]
)
def vis_summary(hi):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gx, y=gy,
        mode='markers',
        marker={'size': 3.5},
        marker_color='rgba(0,128,128,0.8)', name='Normal',
        hovertext=['code=' + str(cdf['code'][i]) for i in range(cdf.shape[0])]
    ))

    fig.add_trace(go.Scatter(
        x=bx, y=by,
        mode='markers',
        marker_color='rgba(128, 0, 0, .6)',
        marker={'size': 2}, name='Risky',
        hovertext=['code=' + str(cdf['code'][i]) for i in range(cdf.shape[0])]
    ))
    fig.update_layout({'legend_orientation': 'h'}, width=400, xaxis_range=[-0.25, 1.25], )

    fig.update_layout(
        xaxis_title="Risky (Binary)",
        yaxis_title="Score",
        font=dict(
            family="Courier New, monospace",
            size=14))

    fig.update_layout(
        title={'text': "Scorecard vs Label", 'y': 1, 'x': 0.5,
               'xanchor': 'center',
               'yanchor': 'top'}
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=20))
    return fig


@app.callback(
    Output('3D', 'figure'), [Input("model-selection", "value")]
)
def visualize_graph(hi):
    fig = px.scatter_3d(cdf, x="Tree_pred", y="Score", z="risk", symbol='industry',
                        hover_data=['code'], color='risk',
                        color_continuous_scale='greens')
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.update_layout({'legend_orientation': 'h'})
    fig.update_traces(showscale=False, selector=dict(type='isosurface'))
    return fig


if __name__ == '__main__':
    app.run_server(debug=False)
# app.run_server(mode='inline')