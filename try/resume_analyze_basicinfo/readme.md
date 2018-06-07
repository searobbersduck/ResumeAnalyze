# 基于LSTM+CRF模型的简历解析方案

## 数据准备
1. 利用idmg的简历提取接口生成简历训练文件，目标代码`idmg/resume_extractor/src/java/com/inmind/idlg/serving/BatchResumeProcessor.java`
    * 生成命令：``bazel-bin/resume_extractor/BatchResumeProcessor resume_extractor/config.json /Users/higgs/Documents/resumes /Users/higgs/Documents/resumes_output``
        * `/Users/higgs/Documents/resumes_output`为用户自己的目录，用来存放用来生成数据集的源文件
    * 对应每份简历源文件（各种格式*.doc，*.pdf等），生成三个文件：
        * *.format_json.txt: c++端得到的简历标注的json
        * *.format_origin.txt：简历经过分段后的json文件
        * *.format_origin_text.txt
2. 生成中文字符词典和中文字符向量
    * 从1中将，简历文本（所有的*.format_origin_text.txt）提取到一个文件（result.txt）
    * 将简历文本（result.txt）分割成一个个字符组成的文本（result_seg.txt），中间以空格隔开，参考代码(./extsrc/seg_char.py)：

        ```python
       import os
       
       lines_list = []
       
       with open('result.txt', 'r') as f:
           for line in f.readlines():
               if not line:
                   continue
               line = line.strip()
               if line is None:
                   continue
               ss = line.split('\t')
               for s in ss:
                   tss = s.split('\x01')
                   if len(tss) < 2:
                       continue
                   tss = tss[1:]
                   for ts in tss:
                       str_line = ''
                       uts = ts.decode('utf8')
                       for i in range(len(uts)):
                           str_line += uts[i]
                           str_line += u' '
                       lines_list.append(str_line)
       
       with open('result_seg.txt','w') as f:
           for line in lines_list:
               f.write(line.encode('utf8'))
               f.write('\n')
        ```
    
    * 利用简历文本（result_seg.txt）提取词汇表：
        * `bazel-bin/third_party/word2vec/word2vec -train result_seg.txt -save-vocab ttt_vocab.txt`
    * 利用简历文本（result_seg.txt）训练词向量：
        * `bazel-bin/third_party/word2vec/word2vec -train result_seg.txt -output ttt_vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -iter 3 -cbow 0`
3. 生成tag词汇表
    * 提取tag词汇表（代码参见`./try/resume_analyze_basicinfo/gen_tfrecord_try.py`中的`test_get_tags_from_files()`函数）：
        * 先从标签文件（basic_keyword.txt和生成的*.format_json.txt）中提取基本的标签
        * 然后在基础标签的基础上生成分词标签（参考：`./try/resume_analyze_basicinfo/gen_tfrecord_try.py`中的`get_tags_all(in_vocab_tag, out_vocab_tag_all)`函数）
4. 统计简历文本的长度，用于设置模型中的输入序列的最大长度：
    * 参考代码`./try/resume_analyze_basicinfo/gen_tfrecord_try.py`中的`stat_max_len_of_resumes()`函数
5. 生成tensorflow使用的数据格式
    * 参见代码：`./try/resume_analyze_basicinfo/gen_tfrecord_try.py`中的`generate_tfrecord()`函数

## 训练模型：
1. 模型文件：`./try/resume_analyze_basicinfo/model_tf.py`
2. 训练代码文件：`./try/resume_analyze_basicinfo/resume_extract_tf.py`
    * 训练命令位于`./try/resume_analyze_basicinfo/task.sh`
    * 命令参数如下：
        ```bash
              optional arguments:
        -h, --help            show this help message and exit
        --train_file TRAIN_FILE
        --val_file VAL_FILE
        --test_file TEST_FILE
        --model {bilstm,bilstm-crf}
        --lr LR
        --mom MOM
        --wd WD
        --iternum ITERNUM
        --phase {train,val,test}
        --output OUTPUT
        --char_vector_file CHAR_VECTOR_FILE
        --char_vocab_file CHAR_VOCAB_FILE
        --tag_vocab_file TAG_VOCAB_FILE
        --hidden_size HIDDEN_SIZE
        --optimizer {sgd,adam}
        --weights WEIGHTS
        --tfdata TFDATA
        --log_dir LOG_DIR
        --batch_size BATCH_SIZE
        --track_history TRACK_HISTORY

        ```
## 训练结果：

|简历数量|训练集|测试集|测试集上准确率|
|:---:|---|---|---|
|2232|80%|20%|98.4%|

## 说明：
1. 目前的结果是提取`*.format_origin.txt`中`basic`标签下的内容组成的文本，即训练的是基本信息相关的内容
2. 由于目前简历源的限制，所用简历都是wuyou的简历
3. 后续需要加大样本的多样性


