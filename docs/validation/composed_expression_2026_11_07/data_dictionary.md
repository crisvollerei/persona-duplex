# 字段、分母与证据

## results.json

- primary_outputs=108：18条件×6轮；verification_outputs=108：对应的不中断机械重放，不是新样本。
- api_calls=0：生成器提供拒绝任何模型调用的函数，归档账本也没有模型调用事件。
- records：case_id/profile/method标识场景、参数组合和表达实现；restart_equal比较整条不中断与重启序列。
- turns：user/sources/frame为实际输入，say为输出，decision为既有程序动作和关注，state为已确认状态。
- expression_plan：composed路径的版本、风格、句式ID、轮次键、最近交付ID、parts；固定方法为null。
- parts：kind=atom为代码内句子片段，quote为归属原文，commitment为应用确认的承诺文本。id及source_id用于追溯；原文不是事实认证。
- signature：把引文和承诺文字替换为类型占位符的新撰结构，用于相邻文字比较；不是语义向量或人格分数。
- delivery_match：回调累计内容与say是否一致；calls_count为本轮实际模型调用数。

## runtime_ledgers.json.gz

按条件ID组织的原始追加账本，含两次run_start、已确认边界/承诺、六次输入/完成及组合方法的预交付计划。运行ID和实际写入时间保留；研究控制时钟只用于策略状态，不混同实际经过时间。

## 编码

review_inputs只有匿名ID、用户文字、资料、开放承诺和完整六轮回复；review_key负责映射，编码任务未获准读取。两份ratings独立保存，不互相修订。turns中的advice与unsupported是逐轮标签，其余五项是整段1—5分，不得当作逐轮评分。reason保存解释，不用于自动修订输出。

## summary.json

- methods：每方法9条件54主输出，旧方法只有3份不同对话，新方法9份。
- coders：两套标签分别汇总；all为9条件平均，分profile为3场景平均。旧配置副本重复，不代表额外独立证据。
- agreement：数值或类别完全相同的项数，逐段分母18、逐轮分母108，不是统计独立性或外部效度。
- adjacent_same_signature：每方法45次条件内相邻比较；旧方法去重后9/15，新方法0/45。
- mean_chars：Python len的Unicode字符计数，不是token。
- audit_report中的790项为材料关系检查，不是790个用户任务；不包含对自然度标签的真值认证。
