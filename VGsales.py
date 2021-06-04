#!usr/bin/env python
# coding:utf-8

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly as py
from plotly.offline import init_notebook_mode, plot
#init_notebook_mode(connected=True)
import plotly.graph_objs as go
from plotly.tools import FigureFactory as ff # 后序话散点矩阵图
import matplotlib.pyplot as plt
import seaborn as sns

import warnings # 忽略warnings
warnings.filterwarnings("ignore")

# 读取数据
vgsales = pd.read_csv("vgsales.csv")

# 首先让我们先来看看该数据集的头和尾，下表显示了数据集开头的前4行
d = vgsales.head(5)
colorscale = "YlOrRd"
table1 = ff.create_table(d, colorscale=colorscale)
for i in range(len(table1.layout.annotations)):
    table1.layout.annotations[i].font.size = 9
    table1.layout.annotations[i].font.color = 'black'
plot(table1)

# 尾30行
t = vgsales.tail(30)
colorscale = "YlOrRd"
table2 = ff.create_table(t, colorscale=colorscale)
for i in range(len(table2.layout.annotations)):
    table2.layout.annotations[i].font.size = 9
    table2.layout.annotations[i].font.color = 'black'
plot(table2)


# 让我们先来看看数据集中各属性的数据类型以及缺失值情况
print("=" * 70)
print(vgsales.info())
print("=" * 70)

print("=" * 70)
print(vgsales.isnull().sum()) #缺失值情况
print("=" * 70)

# 接下来我们看一下数据集中各属性的五数概括：
de = vgsales.describe()
colorscale = "YlOrRd"
table3 = ff.create_table(de, colorscale=colorscale)
for i in range(len(table3.layout.annotations)):
    table3.layout.annotations[i].font.size = 9
    table3.layout.annotations[i].font.color = 'black'
plot(table3)


# 为了能够更好的可视化展示，这里我们对缺失值进行一定的处理
# 这里采用年份平均数填充缺失的年份
def impute_median(series):
    return series.fillna(series.median())
vgsales.Year = vgsales['Year'].transform(impute_median)

# 这里给出整个数据集中发布游戏最多的游戏发布商：
print("=" * 70)
print(vgsales['Publisher'].mode())
print("=" * 70)
# 这里采用发布商众数填充缺失的发布商
vgsales['Publisher'].fillna(str(vgsales['Publisher'].mode().values[0]),inplace=True)

# 检查缺失值填充情况
print("=" * 70)
print(vgsales.isnull().sum())
print("=" * 70)

# 填充完成后让我们再来看一下此时数据集中各属性的五数概括：
de = vgsales.describe()
colorscale = "YlOrRd"
table3 = ff.create_table(de, colorscale=colorscale)
for i in range(len(table3.layout.annotations)):
    table3.layout.annotations[i].font.size = 9
    table3.layout.annotations[i].font.color = 'black'
plot(table3)


### 在进行预测任务前，我将分析视频游戏的销售情况进行可视化展示
### 首先，我想从全球的角度来研究售出最多的100款游戏
### 然后根据不同地区的游戏类型，带有游戏名称的词云，以及最畅销的100款游戏的发行年份和发行商来研究游戏的类型和平台
### 同时也提供了一些关于游戏、发行商和平台的信息

# 首先我们来看一下排名前100的游戏情况，下表展示了排名前100的游戏在不同地区的销售情况：
df = vgsales.head(100)
trace1 = go.Scatter(x=df.Rank, y=df.NA_Sales, mode="markers", name="North America",
                    marker=dict(color='rgba(28, 149, 249, 0.8)', size=8),
                    text=df.Name)
trace2 = go.Scatter(x=df.Rank, y=df.EU_Sales, mode="markers", name="Europe",
                    marker=dict(color='rgba(249, 94, 28, 0.8)', size=8),
                    text=df.Name)
trace3 = go.Scatter(x=df.Rank, y=df.JP_Sales, mode="markers", name="Japan",
                    marker=dict(color='rgba(150, 26, 80, 0.8)', size=8),
                    text=df.Name)
trace4 = go.Scatter(x=df.Rank, y=df.Other_Sales, mode="markers", name="Other",
                    marker=dict(color='lime', size=8),
                    text=df.Name)

data = [trace1, trace2, trace3, trace4]
layout = dict(title='North America, Europe, Japan and Other Sales of Top 100 Video Games',
              xaxis=dict(title='Rank', ticklen=5, zeroline=False, zerolinewidth=1, gridcolor="white"),
              yaxis=dict(title='Sales(In Millions)', ticklen=5, zeroline=False, zerolinewidth=1, gridcolor="white", ),
              paper_bgcolor='rgb(243, 243, 243)',
              plot_bgcolor='rgb(243, 243, 243)')
fig = dict(data=data, layout=layout)
plot(fig)


# 图中可以看到排名前100的游戏在北美的销售额要普遍高于其余地区，这可能与其在不同地域的发布时间有关
# 所以下面我们来看一下游戏发布时间与游戏发布数量的具体情况：
year_count = df.groupby('Year', axis=0).count().reset_index()[['Year','Name']]
year_count.Year = year_count.Year.astype('int')
# 去掉2016年后的发布情况
year_count = year_count[year_count.Year <= 2016]

trace = go.Scatter(x=year_count.Year, y=year_count.Name, mode='lines', name='lines')
layout = go.Layout(title='Release by Year', yaxis=dict(title='Count'), xaxis=dict(title='Year'))

fig = go.Figure(data=[trace], layout=layout)
plot(fig)


# 接下来我们来看一下排名前几位的游戏情况：
print("=" * 70)
print(vgsales.head(10))
print("=" * 70)

# 根据上表可以知道，Wii sports占据了第一的位置，尤其是在北美的销量
# 第二名是《超级马里奥兄弟》，这是任天堂开发并发布的一款平台游戏
# 根据散点图可以看到排名前100的游戏在北美的销售额要普遍高于其余地区，
# 那么这里也列出了一些特殊情况，除了下面列出的那些游戏外，其他大部分都是在北美销售额更高(具体见分析报告):
print("=" * 70)
print(df.iloc[[17, 47]])
print("=" * 70)

print("=" * 70)
print(df.iloc[[10, 16, 19, 34, 37, 54, 77, 82, 83, 92, 93]])
print("=" * 70)

print("=" * 70)
print(df.iloc[[26, 41, 66, 73, 76, 87]])
print("=" * 70)


# 现在展示根据全球销量和游戏发行商来盘点一下前100名游戏的发行年份：
fig = {
    "data": [
        {'x': df.Rank, 'y': df.Year, 'mode': 'markers',
         'marker': {"color": df.Global_Sales, 'size': df.Global_Sales,
                    'showscale': True, "colorscale": 'Blackbody'},
         "text": "Name:" + df.Name + "," + " Publisher:" + df.Publisher
         },
    ],
    "layout":
        {"title": "Release Years of Top 100 Video Games According to Global Sales",
         "xaxis": {"title": "Rank", "gridcolor": 'rgb(255, 255, 255)', "zerolinewidth": 1,
                   "ticklen": 5, "gridwidth": 2,
                   },
         "yaxis": {"title": 'Years', "gridcolor": 'rgb(255, 255, 255)', "zerolinewidth": 1,
                   "ticklen": 5, "gridwidth": 2,
                   },
         "paper_bgcolor": 'rgb(243, 243, 243)',
         "plot_bgcolor": 'rgb(243, 243, 243)'
         }
}

plot(fig)

# 关注与游戏发行商时，我们看到前15款游戏都是由任天堂开发并发行的，下面的图表中也展示了前100款游戏的发行商数量
trace = go.Histogram(x=df.Publisher,
                     marker=dict(color="crimson", line=dict(color='black', width=2)),
                     opacity=0.75)
layout = go.Layout(title='Numbers of Top 100 Video Games Publishers',
                   xaxis=dict(
                       title='Publishers'
                   ),
                   yaxis=dict(
                       title='Count'
                   ),
    bargap=0.2,
    bargroupgap=0.1, paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor="rgb(243, 243, 243)")

fig = go.Figure(data=[trace], layout=layout)
plot(fig)


# 下面的散点图矩阵根据年份，平台，北美，欧洲和全球的销售额之间的关系显示了排名前100的电子游戏:
# 首先我们先看一下排名前100的电子游戏的各属性的五数概括：
print("=" * 70)
print(df.describe())
print("=" * 70)

df2 = df.loc[:, ["Year", "Platform", "NA_Sales", "EU_Sales", "Global_Sales"]]
df2["index"] = np.arange(1, len(df)+1)

# 散点图矩阵
fig = ff.create_scatterplotmatrix(df2, diag='box', index='index', colormap='YlOrRd',
                                  colormap_type='seq', height=1000, width=1200)
plot(fig)



### 接下来从类型和平台的角度来看看电子游戏的销售情况：

# 首先来看看游戏类型和游戏平台的关系
platGenre = pd.crosstab(vgsales.Platform, vgsales.Genre)
platGenreTotal = platGenre.sum(axis=1).sort_values(ascending = False)

trace = go.Bar(x=platGenreTotal.index, y=platGenreTotal.values,
               marker=dict(color="rgb(10,77,131)"), opacity = 0.75)
layout = go.Layout(title='The Relation Between Game Platform And Genre',
                   xaxis=dict(title='Platform'),
                   yaxis=dict(title='The amount of games'),
                   bargap=0.2,
                   bargroupgap=0.1,
                   paper_bgcolor='rgb(243, 243, 243)',
                   plot_bgcolor="rgb(243, 243, 243)")

fig = go.Figure(data=[trace], layout=layout)
plot(fig)

# 从上图中可以看到DS和PS2在他们的平台上拥有最多的游戏，为了我们可以看到平台上拥有超过1000款游戏的游戏类型的细节
# 接下来我们分别采用堆叠条的直方图和热力图来展示游戏类型细节：
# 分别统计各类型的游戏
xaction = vgsales[vgsales.Genre == "Action"]
xsports = vgsales[vgsales.Genre == "Sports"]
xmisc = vgsales[vgsales.Genre == "Misc"]
xrole = vgsales[vgsales.Genre == "Role-Playing"]
xshooter = vgsales[vgsales.Genre == "Shooter"]
xadventure = vgsales[vgsales.Genre == "Adventure"]
xrace = vgsales[vgsales.Genre == "Racing"]
xplatform = vgsales[vgsales.Genre == "Platform"]
xsimulation = vgsales[vgsales.Genre == "Simulation"]
xfight = vgsales[vgsales.Genre == "Fighting"]
xstrategy = vgsales[vgsales.Genre == "Strategy"]
xpuzzle = vgsales[vgsales.Genre == "Puzzle"]

# 将各类型游戏与平台的关系画在同一图中
trace1 = go.Histogram(x=xaction.Platform, opacity=0.75, name="Action",
                      marker=dict(color='rgb(165,0,38)'))
trace2 = go.Histogram(x=xsports.Platform, opacity=0.75, name="Sports",
                      marker=dict(color='rgb(215,48,39)'))
trace3 = go.Histogram(x=xmisc.Platform, opacity=0.75, name="Misc",
                      marker=dict(color='rgb(244,109,67)'))
trace4 = go.Histogram(x=xrole.Platform, opacity=0.75, name="Role Playing",
                      marker=dict(color='rgb(253,174,97)'))
trace5 = go.Histogram(x=xshooter.Platform, opacity=0.75, name="Shooter",
                      marker=dict(color='rgb(254,224,144)'))
trace6 = go.Histogram(x=xadventure.Platform, opacity=0.75, name="Adventure",
                      marker=dict(color='rgb(170,253,87)'))
trace7 = go.Histogram(x=xrace.Platform, opacity=0.75, name="Racing",
                      marker=dict(color='rgb(171,217,233)'))
trace8 = go.Histogram(x=xplatform.Platform, opacity=0.75, name="Platform",
                      marker=dict(color='rgb(116,173,209)'))
trace9 = go.Histogram(x=xsimulation.Platform, opacity=0.75, name="Simulation",
                      marker=dict(color='rgb(69,117,180)'))
trace10 = go.Histogram(x=xfight.Platform, opacity=0.75, name="Fighting",
                       marker=dict(color='rgb(49,54,149)'))
trace11 = go.Histogram(x=xstrategy.Platform, opacity=0.75, name="Strategy",
                       marker=dict(color="rgb(10,77,131)"))
trace12 = go.Histogram(x=xpuzzle.Platform, opacity=0.75, name="Puzzle",
                       marker=dict(color='rgb(1,15,139)'))

data = [trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9, trace10, trace11, trace12]
layout = go.Layout(barmode='stack', title='Genre Counts According to Platform',
                   xaxis=dict(title='Platform'), yaxis=dict( title='Count'),
                   paper_bgcolor='beige', plot_bgcolor='beige'
)
fig = go.Figure(data=data, layout=layout)
plot(fig)

# 上面的图表展示了数据集中基于发布平台数量的游戏类型
# 接着我们用更直观的热力图来进行展示：
platGenre['Total'] = platGenre.sum(axis=1)
popPlatform = platGenre[platGenre['Total']>1000].sort_values(by='Total', ascending = False)
neededdata = popPlatform.loc[:, :'Strategy']
maxi = neededdata.values.max()
mini = neededdata.values.min()
popPlatformfinal = popPlatform.append(pd.DataFrame(popPlatform.sum(), columns=['total']).T, ignore_index=False)

sns.set(font_scale=2)
plt.figure(figsize=(30, 20))
sns.heatmap(popPlatformfinal, vmin=mini, vmax=maxi, annot=True, fmt="d", cmap="YlOrRd")
plt.xticks(rotation=90)
plt.ylabel("Platform")
plt.show()


# 从上面的直方图和热力图可以看出PS3平台上的动作类游戏，DS平台上的音乐类游戏以及PS2平台上的运动类游戏是最受欢迎的
# 接下来我们看一下基于平台和游戏类型的全球销售额情况：
trace1 = go.Bar(x=xaction.groupby("Platform")["Global_Sales"].sum().index,
                y=xaction.groupby("Platform")["Global_Sales"].sum().values,
                opacity=0.75, name="Action", marker=dict(color="rgb(119,172,238)"))
trace2 = go.Bar(x=xsports.groupby("Platform")["Global_Sales"].sum().index,
                y=xsports.groupby("Platform")["Global_Sales"].sum().values,
                opacity=0.75, name="Sports", marker=dict(color='rgb(21,90,174)'))
trace3 = go.Bar(x=xrace.groupby("Platform")["Global_Sales"].sum().index,
                y=xrace.groupby("Platform")["Global_Sales"].sum().values,
                opacity=0.75, name="Racing", marker=dict(color="rgb(156,245,163)"))
trace4 = go.Bar(x=xshooter.groupby("Platform")["Global_Sales"].sum().index,
                y=xshooter.groupby("Platform")["Global_Sales"].sum().values,
                opacity=0.75, name="Shooter", marker=dict(color="rgb(14,135,23)"))
trace5 = go.Bar(x=xmisc.groupby("Platform")["Global_Sales"].sum().index,
                y=xmisc.groupby("Platform")["Global_Sales"].sum().values,
                opacity=0.75, name="Misc", marker=dict(color='rgb(252,118,103)'))
trace6 = go.Bar(x=xrole.groupby("Platform")["Global_Sales"].sum().index,
                y=xrole.groupby("Platform")["Global_Sales"].sum().values,
                opacity=0.75, name="Role Playing", marker=dict(color="rgb(226,28,5)"))
trace7 = go.Bar(x=xfight.groupby("Platform")["Global_Sales"].sum().index,
                y=xfight.groupby("Platform")["Global_Sales"].sum().values,
                opacity=0.75, name="Fighting", marker=dict(color="rgb(247,173,13)"))
trace8 = go.Bar(x=xplatform.groupby("Platform")["Global_Sales"].sum().index,
                y=xplatform.groupby("Platform")["Global_Sales"].sum().values,
                opacity=0.75, name="Platform", marker=dict(color="rgb(242,122,13)"))
trace9 = go.Bar(x=xsimulation.groupby("Platform")["Global_Sales"].sum().index,
                y=xsimulation.groupby("Platform")["Global_Sales"].sum().values,
                opacity=0.75, name="Simulation", marker=dict(color="rgb(188,145,202)"))
trace10 = go.Bar(x=xadventure.groupby("Platform")["Global_Sales"].sum().index,
                 y=xadventure.groupby("Platform")["Global_Sales"].sum().values,
                 opacity=0.75, name="Adventure", marker=dict(color='rgb(104,57,119)'))
trace11 = go.Bar(x=xstrategy.groupby("Platform")["Global_Sales"].sum().index,
                 y=xstrategy.groupby("Platform")["Global_Sales"].sum().values,
                 opacity=0.75, name="Strategy", marker=dict(color='rgb(245,253,104)'))
trace12 = go.Bar(x=xpuzzle.groupby("Platform")["Global_Sales"].sum().index,
                 y=xpuzzle.groupby("Platform")["Global_Sales"].sum().values,
                 opacity=0.75, name="Puzzle", marker=dict(color='rgb(138,72,40)'))

data = [trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9, trace10, trace11, trace12]
layout = go.Layout(barmode='stack', title='Total Global Sales According to Platform and Genre',
                   xaxis=dict(title='Platform'), yaxis=dict( title='Global Sales(In Millions)'),
                   paper_bgcolor='beige', plot_bgcolor='beige'
                   )

fig = go.Figure(data=data, layout=layout)
plot(fig)



### 现在根据类型和地区来分析销售情况：
# 首先我们来看一下不同类型的游戏的全球销售额：
sales_by_genre = vgsales.groupby(['Genre', 'Name'], axis=0).sum().reset_index()[['Genre', 'Name', 'Global_Sales']]
print("=" * 70)
print(sales_by_genre.head(10))
print("=" * 70)
genres = sales_by_genre.Genre.unique()
traces = []
c = ['hsl(' + str(h) + ',50%' + ',50%)' for h in np.linspace(0, 360, len(genres))]

for i in range(len(genres)):
    genre = genres[i]
    df_genre = sales_by_genre[sales_by_genre.Genre == genre]
    trace = go.Box(y=np.array(df_genre.Global_Sales), name=genre,
                   boxmean=True, marker={'color': c[i]})
    traces.append(trace)

layout = go.Layout(title='Global_Sales by Genre (Overall dataset)', showlegend=False,
                   yaxis=dict(title='Sales (in Millions)'), xaxis=dict(title='Genre'))

fig = go.Figure(data=traces, layout=layout)
plot(fig)

# 接着我们单独来看看排名前100的不同类型的游戏在北美的销售额：
df = vgsales.head(100)
sales_by_genre = df.groupby(['Genre', 'Name'], axis=0).sum().reset_index()[['Genre', 'Name', 'NA_Sales']]
print("=" * 70)
print(sales_by_genre.head(10))
print("=" * 70)
genres = sales_by_genre.Genre.unique()
traces = []
c = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, len(genres))]

for i in range(len(genres)):
    genre = genres[i]
    df_genre = sales_by_genre[sales_by_genre.Genre == genre]
    trace = go.Box(y=np.array(df_genre.NA_Sales), name=genre,
                   boxmean=True, marker={'color': c[i]})
    traces.append(trace)

layout = go.Layout(title='NA_Sales by Genre (Top 100)', showlegend=False,
                   yaxis=dict(title='Sales (in Millions)'), xaxis=dict(title='Genre'))

fig = go.Figure(data=traces, layout=layout)
plot(fig)

# 为了能够直观的看到各个类型的游戏在不同地区的销售额的情况，将他们统计后放在一张图中：
# 这里将Genre作为groupby依据，统计各类型游戏在各地区的销售额
genre = pd.DataFrame(vgsales.groupby("Genre")
                     [["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]].sum())
genre.reset_index(level=0, inplace=True)
genrecount = pd.DataFrame(vgsales["Genre"].value_counts())
genrecount.reset_index(level=0, inplace=True)
genrecount.rename(columns={"Genre": "Counts", "index": "Genre"}, inplace=True)

genre = pd.merge(genre, genrecount, on="Genre")

table_data = genre[["Genre", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]]
table_data = table_data.rename(columns = {"NA_Sales": "North America", "EU_Sales": "Europe",
                                  "JP_Sales": "Japan", "Other_Sales": "Other", "Global_Sales": "Total"})

x = genre.Genre
# 计算不同类型游戏在各地区销售额占总销售额的百分比
NA_Perce = list(genre["NA_Sales"] / genre["Global_Sales"] * 100)
EU_Perce = list(genre["EU_Sales"] / genre["Global_Sales"] * 100)
JP_Perce=list(genre["JP_Sales"] / genre["Global_Sales"] * 100)
Other_Perce=list(genre["Other_Sales"] / genre["Global_Sales"] * 100)


# 下面左边的图表向我们展示了不同地区的游戏类型百分比。例如，大约50%的动作类电影是在北美销售的，
# 同时可以在右边的表中看到在北美销售的具体值
# 例如，模拟类型游戏的总销售额为3.92亿。其中6400万是在日本售出的，总体来看，当关注于游戏类型时，我们会发现：
# 北美地区的销售比例在44-56%之间（不包括角色扮演和策略类型的游戏），销售最多的类型是射击游戏和平台游戏
# 在日本，主要销售的是角色扮演类游戏，而射击类游戏的销量低于其他地区
# 在欧洲，整体看来各类游戏的销售额所占百分比较为平均，各类游戏都受到青睐
# 在其他地区，销售额较多的游戏类型是竞速、动作和运动，并且各类游戏的销售额所占百分比都较小

# 画Bar图
trace1 = go.Bar(x=x, y=NA_Perce, name="North America" , xaxis='x2', yaxis='y2',
                marker=dict(color='rgb(158,202,225)', line=dict(color='rgb(8,48,107)', width=3)),
                opacity=0.75)
trace2 = go.Bar(x=x, y=EU_Perce, name="Europe", xaxis='x2', yaxis='y2',
                marker=dict(color='red', line=dict(color='rgb(8,48,107)',width=3)),
                opacity=0.75)
trace3 = go.Bar(x=x, y=JP_Perce, name="Japan", xaxis='x2', yaxis='y2',
                marker=dict(color='orange', line=dict(color='rgb(8,48,107)',width=3)),
                opacity=0.75)
trace4 = go.Bar(x=x, y=Other_Perce, name="Other", xaxis='x2', yaxis='y2',
                marker=dict(color='purple', line=dict(color='rgb(8,48,107)',width=3)),
                opacity=0.75)

# 画表
trace5 = go.Table(
    header=dict(values=table_data.columns, line=dict(color='rgb(8,48,107)', width=3),
                  fill=dict(color=["darkslateblue", "blue", "red", "orange", "purple", "green"]),
                  align=['left', 'center'], font=dict(color='white', size=12), height=30,),
    cells=dict(values=[table_data.Genre, round(table_data["North America"]),
                       round(table_data["Europe"]), round(table_data["Japan"]),
                       round(table_data["Other"]), round(table_data["Total"])],
                 height=30, line=dict(color='rgb(8,48,107)', width=3),
                 fill=dict(color=["silver", "rgb(158,202,225)", "darksalmon",
                                  "gold", "mediumorchid", "yellowgreen"]),
                 align=['left', 'center'], font=dict(color='#506784', size = 12)),
    domain=dict(x=[0.60, 1], y=[0, 0.95]))

data = [trace1, trace2, trace3, trace4, trace5]
layout = go.Layout(barmode='stack', autosize=False, width=1200, height=650,
                legend=dict(x=.58, y=0, orientation="h",
                            font=dict(family='Courier New, monospace', size=11, color='#000'),
                            bgcolor='beige', bordercolor='beige', borderwidth=1),
                title='North America, Europe, Japan and Other Sales Percentage and Amounts According to Genre',
                titlefont=dict(family='Courier New, monospace', size=17, color='black'),
                xaxis2=dict(domain=[0, 0.50], anchor="y2", title='Genre',
                            titlefont=dict(family='Courier New, monospace'),
                            tickfont=dict(family='Courier New, monospace')
                            ),
                yaxis2=dict(domain=[0, 1], anchor='x2', title="Total Percentage",
                            titlefont=dict(family='Courier New, monospace'),
                            tickfont=dict(family='Courier New, monospace')
                            ),
                paper_bgcolor='beige', plot_bgcolor='beige',
                annotations=[dict(text='Sales Percentage According to Region',
                                  x=0.08, y=1.02, xref="paper", yref="paper",
                                  showarrow=False, font=dict(size=15,family="Courier New, monospace"),
                                  bgcolor="lightyellow",borderwidth=5),
                             dict(text='Total Sales(In Millions)', x=0.9, y=1.02,
                                  xref="paper", yref="paper", showarrow=False,
                                  font=dict(size=15, family='Courier New, monospace'),
                                  bgcolor="lightyellow", borderwidth=5)])

fig = go.Figure(data=data, layout=layout)
plot(fig)