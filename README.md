# 吃饱喝足 Bot 使用说明
### 环境配置
配置一个 `python>=3.7` 的环境, 并安装 `requirements.txt` 中列出的库
推荐配置方式为使用 Anaconda, 输入如下命令完成配置:
```shell
conda create -n [your_env_name] python=3.7
conda activate [your_env_name]
pip install -r requirements.txt
```
### 模型使用
执行 `python interact.py` 与 Bot 进行交互. 可指定参数如下:
- `--device`: 计算设备, 默认为 cpu;
- `--model_dir`: 模型存放目录, 默认为 `./eat_and_drink_full_bot_model/`;
- `--max_len`: 生成文本的最大长度. 超过指定长度则进行截断, 默认值为 64;
- `--max_history_len`: 生成文本时考虑的前文对话轮数, 默认值为 3.
- `--temperature`: 生成时的 temperature. 值越高所生成文本的随机性越强, 默认值为 1;
- `--repetition_penalty`: 重复惩罚参数. 若生成的对话重复性较高, 可适当提高该参数, 默认值为 1.5;
- `--topk`: 解码算法 `top_k_top_p_filtering` 的参数 `k`, 从概率最高的 `k` 个 token 中选择. 若 `k==1` 即为贪心搜索. 默认值为 8;
- `--topp`: 解码算法 `top_k_top_p_filtering` 的参数 `p`, 从概率最高且概率之和恰好超过 `p` 的若干个 token 中选择. 若 `p==0.0` 则无效. 默认值为 0.0;

每输入一段以换行符结尾的文本后, Bot 就会读取输入并回复一句话. 
输入 `break` 或 `end` 结束对话.