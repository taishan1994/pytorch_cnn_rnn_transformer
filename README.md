# pytorch_cnn_rnn_transformer
pytorch版本的三大特征提取器：BILSTM、CNN、Transformer

# 说明
该项目提供通用的三大特征提取器模块，旨在可以对句子进行快速的建模。只提供了分类的示例，很容易迁移到其它的任务上。<br>其中：使用的输入是以字为单位的，没有使用预训练的字向量。字表是基于ChineseBert里面的vocab.txt，提取了里面长度为1的词组成的。CNN和BILSTM是从sentence-transformers里面截取出来的。Transformer是pytoch官方自带的，根据官方的tutorial进行改写，具体为改写输入为[batchsize, max_seq_len]格式，改写位置嵌入，仅使用编码层进行分类。使用的数据是短文本，进行10分类。项目结构：<br><br>
--checkpoints：模型存储位置，里面有cnn、rnn、transformer。<br>
--logs：训练、验证、测试和预测的日志<br>
--data：数据<br>
--cnn.py<br>
--rnn.py<br>
--transformer.py<br>
--utils.py：设置随机种子以及日志模块<br>

# 依赖
```python
sklearn
pytorch==1.6.0+
```

# 执行指令
注意里面的训练、验证、测试和预测代码是否注释掉。
```python
python cnn.py or python lstm.py or python transformer.py
```

# 结果
| 模型      | f1 |
| ----------- | ----------- |
| cnn      | 0.89       |
| rnn   | 0.90        |
| transformer   | 0.90        |

#测试
以transformer为例：
```python             
				precision    recall  f1-score   support

           0       0.87      0.89      0.88      1000
           1       0.91      0.92      0.92      1000
           2       0.84      0.83      0.84      1000
           3       0.96      0.92      0.94      1000
           4       0.84      0.84      0.84      1000
           5       0.89      0.92      0.90      1000
           6       0.89      0.87      0.88      1000
           7       0.96      0.97      0.97      1000
           8       0.92      0.92      0.92      1000
           9       0.93      0.92      0.92      1000

    accuracy                           0.90     10000
   macro avg       0.90      0.90      0.90     10000
weighted avg       0.90      0.90      0.90     10000
```

# 测试
```python
2021-09-12 18:56:19,757 - INFO - transformer.py - <module> - 320 - ======================================
2021-09-12 18:56:19,757 - INFO - transformer.py - <module> - 321 - 词汇阅读是关键 08年考研暑期英语复习全指南
2021-09-12 18:56:19,762 - INFO - transformer.py - <module> - 323 - 真实标签：3
2021-09-12 18:56:19,763 - INFO - transformer.py - <module> - 324 - 预测标签：3
2021-09-12 18:56:19,763 - INFO - transformer.py - <module> - 320 - ======================================
2021-09-12 18:56:19,763 - INFO - transformer.py - <module> - 321 - 中国人民公安大学2012年硕士研究生目录及书目
2021-09-12 18:56:19,764 - INFO - transformer.py - <module> - 323 - 真实标签：3
2021-09-12 18:56:19,764 - INFO - transformer.py - <module> - 324 - 预测标签：3
2021-09-12 18:56:19,764 - INFO - transformer.py - <module> - 320 - ======================================
2021-09-12 18:56:19,765 - INFO - transformer.py - <module> - 321 - 杨澜公布赴台采访老照片 双颊圆润似董洁(组图)
2021-09-12 18:56:19,766 - INFO - transformer.py - <module> - 323 - 真实标签：9
2021-09-12 18:56:19,766 - INFO - transformer.py - <module> - 324 - 预测标签：9
2021-09-12 18:56:19,766 - INFO - transformer.py - <module> - 320 - ======================================
2021-09-12 18:56:19,766 - INFO - transformer.py - <module> - 321 - 梁羽生金庸也曾打过笔战(图)
2021-09-12 18:56:19,768 - INFO - transformer.py - <module> - 323 - 真实标签：9
2021-09-12 18:56:19,768 - INFO - transformer.py - <module> - 324 - 预测标签：9
2021-09-12 18:56:19,768 - INFO - transformer.py - <module> - 320 - ======================================
2021-09-12 18:56:19,768 - INFO - transformer.py - <module> - 321 - 泰达荷银基金：管理层维护市场稳定信心坚决
2021-09-12 18:56:19,770 - INFO - transformer.py - <module> - 323 - 真实标签：0
2021-09-12 18:56:19,770 - INFO - transformer.py - <module> - 324 - 预测标签：0
2021-09-12 18:56:19,770 - INFO - transformer.py - <module> - 320 - ======================================
2021-09-12 18:56:19,770 - INFO - transformer.py - <module> - 321 - 国家意志或影响汇市 对美元暂宜持多头思路
2021-09-12 18:56:19,772 - INFO - transformer.py - <module> - 323 - 真实标签：0
2021-09-12 18:56:19,772 - INFO - transformer.py - <module> - 324 - 预测标签：2
2021-09-12 18:56:19,772 - INFO - transformer.py - <module> - 320 - ======================================
2021-09-12 18:56:19,772 - INFO - transformer.py - <module> - 321 - QQ宝贝零花钱大作战 攒钱拿奖两不误
2021-09-12 18:56:19,774 - INFO - transformer.py - <module> - 323 - 真实标签：8
2021-09-12 18:56:19,774 - INFO - transformer.py - <module> - 324 - 预测标签：8
2021-09-12 18:56:19,774 - INFO - transformer.py - <module> - 320 - ======================================
2021-09-12 18:56:19,774 - INFO - transformer.py - <module> - 321 - 微软：迈向3D游戏还得2-3年时间
2021-09-12 18:56:19,776 - INFO - transformer.py - <module> - 323 - 真实标签：8
2021-09-12 18:56:19,776 - INFO - transformer.py - <module> - 324 - 预测标签：8
2021-09-12 18:56:19,776 - INFO - transformer.py - <module> - 320 - ======================================
2021-09-12 18:56:19,776 - INFO - transformer.py - <module> - 321 - 法国对格鲁吉亚观察员使命未获延期表遗憾
2021-09-12 18:56:19,778 - INFO - transformer.py - <module> - 323 - 真实标签：6
2021-09-12 18:56:19,778 - INFO - transformer.py - <module> - 324 - 预测标签：6
2021-09-12 18:56:19,778 - INFO - transformer.py - <module> - 320 - ======================================
2021-09-12 18:56:19,778 - INFO - transformer.py - <module> - 321 - 媒体报道希拉里出任美国国务卿触发违宪争议
2021-09-12 18:56:19,779 - INFO - transformer.py - <module> - 323 - 真实标签：6
2021-09-12 18:56:19,779 - INFO - transformer.py - <module> - 324 - 预测标签：6
2021-09-12 18:56:19,780 - INFO - transformer.py - <module> - 320 - ======================================
2021-09-12 18:56:19,780 - INFO - transformer.py - <module> - 321 - 朝阳北路低密现房新天际国庆热销备受关注
2021-09-12 18:56:19,781 - INFO - transformer.py - <module> - 323 - 真实标签：1
2021-09-12 18:56:19,781 - INFO - transformer.py - <module> - 324 - 预测标签：1
2021-09-12 18:56:19,781 - INFO - transformer.py - <module> - 320 - ======================================
2021-09-12 18:56:19,782 - INFO - transformer.py - <module> - 321 - 北京四环路以内房价每平方米达17478元
2021-09-12 18:56:19,783 - INFO - transformer.py - <module> - 323 - 真实标签：1
2021-09-12 18:56:19,783 - INFO - transformer.py - <module> - 324 - 预测标签：1
2021-09-12 18:56:19,783 - INFO - transformer.py - <module> - 320 - ======================================
2021-09-12 18:56:19,783 - INFO - transformer.py - <module> - 321 - 13英寸苹果MacBook Pro轻薄时尚本促销
2021-09-12 18:56:19,785 - INFO - transformer.py - <module> - 323 - 真实标签：4
2021-09-12 18:56:19,785 - INFO - transformer.py - <module> - 324 - 预测标签：4
2021-09-12 18:56:19,785 - INFO - transformer.py - <module> - 320 - ======================================
2021-09-12 18:56:19,785 - INFO - transformer.py - <module> - 321 - 瑞星侵权败诉后拒不道歉 法院发公告强制执行
2021-09-12 18:56:19,787 - INFO - transformer.py - <module> - 323 - 真实标签：4
2021-09-12 18:56:19,787 - INFO - transformer.py - <module> - 324 - 预测标签：9
2021-09-12 18:56:19,787 - INFO - transformer.py - <module> - 320 - ======================================
2021-09-12 18:56:19,787 - INFO - transformer.py - <module> - 321 - 图文：安徽发现葫芦形乌龟
2021-09-12 18:56:19,789 - INFO - transformer.py - <module> - 323 - 真实标签：5
2021-09-12 18:56:19,789 - INFO - transformer.py - <module> - 324 - 预测标签：5
2021-09-12 18:56:19,789 - INFO - transformer.py - <module> - 320 - ======================================
2021-09-12 18:56:19,789 - INFO - transformer.py - <module> - 321 - 男子杀死18岁怀孕情人 称怕妻子发现
2021-09-12 18:56:19,791 - INFO - transformer.py - <module> - 323 - 真实标签：5
2021-09-12 18:56:19,791 - INFO - transformer.py - <module> - 324 - 预测标签：5
2021-09-12 18:56:19,791 - INFO - transformer.py - <module> - 320 - ======================================
2021-09-12 18:56:19,791 - INFO - transformer.py - <module> - 321 - 晋级时刻穆帅只留下一个背影 如此国米才配属于他
2021-09-12 18:56:19,793 - INFO - transformer.py - <module> - 323 - 真实标签：7
2021-09-12 18:56:19,793 - INFO - transformer.py - <module> - 324 - 预测标签：7
2021-09-12 18:56:19,793 - INFO - transformer.py - <module> - 320 - ======================================
2021-09-12 18:56:19,793 - INFO - transformer.py - <module> - 321 - 奇才新援10分钟5分5助攻2帽 10天合同竟能捡到宝
2021-09-12 18:56:19,795 - INFO - transformer.py - <module> - 323 - 真实标签：7
2021-09-12 18:56:19,795 - INFO - transformer.py - <module> - 324 - 预测标签：7
2021-09-12 18:56:19,795 - INFO - transformer.py - <module> - 320 - ======================================
2021-09-12 18:56:19,795 - INFO - transformer.py - <module> - 321 - 沪深两市触底反弹 分析人士称中长期仍然看好
2021-09-12 18:56:19,797 - INFO - transformer.py - <module> - 323 - 真实标签：2
2021-09-12 18:56:19,797 - INFO - transformer.py - <module> - 324 - 预测标签：2
2021-09-12 18:56:19,797 - INFO - transformer.py - <module> - 320 - ======================================
2021-09-12 18:56:19,797 - INFO - transformer.py - <module> - 321 - 《经济学人》封面文章：欧洲援助方案
2021-09-12 18:56:19,799 - INFO - transformer.py - <module> - 323 - 真实标签：2
2021-09-12 18:56:19,799 - INFO - transformer.py - <module> - 324 - 预测标签：2
```

# 参考
> https://github.com/ShannonAI/ChineseBert
> https://github.com/UKPLab/sentence-transformers