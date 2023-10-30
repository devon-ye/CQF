import re
import jieba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.font_manager import FontProperties
from sklearn.pipeline import Pipeline
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.feature_selection import SelectKBest, chi2
from collections import Counter

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from hyperopt.pyll.base import scope


pd.set_option('display.max_rows', None)

# 设置中文字体
fontPath = '/System/Library/Fonts/PingFang.ttc'
myFont = FontProperties(fname=fontPath)

# 1. 数据加载
# 加载数据
data = pd.read_csv('../datasets/ChnSentiCorp_htl_all.csv')
# 加载停用词
stopwords = [line.strip() for line in open('../datasets/baidu_stopwords.txt', 'r', encoding='utf-8').readlines()]

# 2. 文本预处理
# 将浮点数转换为保留两位小数的字符串
data['text'] = data['text'].astype(str)


# 定义一个函数来进行文本预处理
def preprocess(text):
    # 去除标点符号和特殊字符
    text = re.sub(r'[^\w\s]', '', text)
    # 分词
    words = jieba.cut(text)
    # 去除停用词
    words = [word for word in words if word not in stopwords]
    return ' '.join(words)


# 对文本进行预处理
data['text'] = data['text'].apply(preprocess)

# 3. 数据探索和可视化
# 计算文本长度分布
data['text_length'] = data['text'].apply(len)

# 设置文本长度阈值
threshold = 500

# 生成颜色数组
colors = np.where(data['text_length'] < threshold, 'blue', 'red')

# 计算文本长度的分布
counts, bins = np.histogram(data['text_length'], bins=200)

# 绘制文本长度分布直方图
plt.figure(figsize=(15, 10))
plt.bar(bins[:-1], counts, width=np.diff(bins), color=colors[:len(bins) - 1], edgecolor='red')
plt.xlabel('文本长度', fontproperties=myFont)
plt.ylabel('频数', fontproperties=myFont)
plt.title('文本长度分布', fontproperties=myFont)
plt.show()

# 绘制标签分布柱状图
plt.figure(figsize=(6, 4))
data['label'].value_counts().plot(kind='bar')
plt.xlabel('标签', fontproperties=myFont)
plt.ylabel('条数', fontproperties=myFont)
plt.title('情绪分布', fontproperties=myFont)
plt.xticks([0, 1], ['积极的', '消极的'], fontproperties=myFont)
plt.show()

# 生成词云
text = ' '.join(data['text'])
wordcloud = WordCloud(width=800, height=400, background_color='white', font_path=fontPath).generate(text)

# 绘制词云
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# 计算词频
word_counts = Counter(' '.join(data['text']).split())
# 绘制词频分布图
top_words = word_counts.most_common(25)
words, counts = zip(*top_words)
# 绘制横向条形图
relative_counts = counts / np.max(counts)
# 绘制横向条形图
plt.figure(figsize=(10, 6))
plt.barh(words, counts, color=plt.cm.Greens(relative_counts))
plt.ylabel('词语', fontproperties=myFont)
plt.xlabel('频次', fontproperties=myFont)
plt.title('词频分布', fontproperties=myFont)
plt.yticks(range(len(words)), words, fontproperties=myFont)
plt.show()

# 4. 特征表示
tfidf_vectorizer = TfidfVectorizer(max_df=0.9,min_df=7,max_features=10000)

# 创建朴素贝叶斯分类器
classifier = MultinomialNB(alpha=0.23)

# 创建卡方特征选择器
kbest = SelectKBest(chi2, k=30)

# 创建管道
pipeline = Pipeline([
    ('tfidf', tfidf_vectorizer),
    ('kbest', kbest),
    ('clf', classifier)
])

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# 5. 模型训练
# 训练模型

# classifier = pipeline.named_steps['clf']
pipeline.fit(X_train, y_train)

# 6. 模型评估
# 对测试集进行预测
y_pred = pipeline.predict(X_test)

# 打印分类报告
print(classification_report(y_test, y_pred))

# 绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('预测标签', fontproperties=myFont)
plt.ylabel('实际标签', fontproperties=myFont)
plt.show()

# 计算 ROC 曲线和 AUC 分数
fpr, tpr, _ = roc_curve(y_test, pipeline.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

# 绘制 ROC 曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC 曲线 (面积 = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('假阳性率', fontproperties=myFont)
plt.ylabel('真阳性率', fontproperties=myFont)
plt.title('接收者操作特征曲线', fontproperties=myFont)
plt.legend(loc='lower right', prop=myFont)
plt.show()

# 7. 特征选择
# 使用卡方校验选择特征
kbest_instance = pipeline.named_steps['kbest']
# 获取选定的特征
selected_features = kbest_instance.get_support(indices=True)

# 获取特征分数
feature_scores = kbest_instance.scores_

# 获取选定的特征的名字
selected_feature_names = np.array(tfidf_vectorizer.get_feature_names_out())[selected_features]

# 获取选定特征的分数
selected_feature_scores = feature_scores[selected_features]

# 绘制条形图
plt.figure(figsize=(10, 6))
sns.barplot(x=selected_feature_scores, y=selected_feature_names)
plt.xlabel('卡方校验值', fontproperties=myFont)
plt.ylabel('特征', fontproperties=myFont)
plt.title('卡方校验特征重要性', fontproperties=myFont)
plt.yticks(fontproperties=myFont)
plt.show()


# 8. 模型调优
# 定义搜索空间
space = {
    'tfidf__min_df': scope.int(hp.quniform('tfidf__min_df', 1, 10, 1)),
    'tfidf__max_df': hp.uniform('tfidf__max_df', 0.5, 1.0),
    'tfidf__max_features': scope.int(hp.quniform('tfidf__max_features', 1000, 3000, 50)),
    'tfidf__ngram_range': hp.choice('tfidf__ngram_range', [(1, 1), (1, 2)]),
    'clf__alpha': hp.uniform('clf__alpha', 0.01, 1),
    'kbest__k': scope.int(hp.quniform('kbest__k', 10, 1000, 10))
}

# 定义目标函数
def objective(params):
    pipeline.set_params(**params)
    score = -np.mean(cross_val_score(pipeline, data['text'], data['label'], cv=5))
    return score

# 进行贝叶斯优化
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)

# 打印最佳参数
print('最佳参数:', best)

# 使用最佳参数对模型进行拟合
best_params = space_eval(space, best)
pipeline.set_params(**best_params)
pipeline.fit(data['text'], data['label'])

# 对测试集进行预测
y_pred = pipeline.predict(X_test)

# 打印分类报告
print(classification_report(y_test, y_pred))

# 绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('预测标签', fontproperties=myFont)
plt.ylabel('实际标签', fontproperties=myFont)
plt.show()

# 计算 ROC 曲线和 AUC 分数
fpr, tpr, _ = roc_curve(y_test, pipeline.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

# 绘制 ROC 曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC 曲线 (面积 = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('假阳性率', fontproperties=myFont)
plt.ylabel('真阳性率', fontproperties=myFont)
plt.title('接收者操作特征曲线', fontproperties=myFont)
plt.legend(loc='lower right', prop=myFont)
plt.show()
