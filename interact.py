import argparse
import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast, GPT2LMHeadModel

class Inference:
    def __init__(
            self,
            model_name_or_path,
            device="cpu",
            max_history_len=3,
            max_len=64,
            repetition_penalty=1.0,
            temperature=1.0,
            topk=8,
            topp=0.0
    ):
        self.device = device
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name_or_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()
        self.history = []
        self.max_history_len = max_history_len
        self.max_len = max_len
        self.repetition_penalty = repetition_penalty
        self.temperature = temperature
        self.topk = topk
        self.topp = topp

    def predict(self, query, use_history=True):
        text_ids = self.tokenizer.encode(query, add_special_tokens=False)
        self.history.append(text_ids)
        input_ids = [self.tokenizer.cls_token_id]
        if use_history:
            for history_id, history_utr in enumerate(self.history[-self.max_history_len:]):
                input_ids.extend(history_utr)
                input_ids.append(self.tokenizer.sep_token_id)
        else:
            input_ids.extend(text_ids)
            input_ids.append(self.tokenizer.sep_token_id)
        input_ids = torch.tensor(input_ids).long().to(self.device)
        input_ids = input_ids.unsqueeze(0)
        response = []
        # 最多生成 max_len 个 token
        for _ in range(self.max_len):
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits
            next_token_logits = logits[0, -1, :]

            # 重复惩罚项, 降低已有 token 的生成概率
            for id in set(response):
                next_token_logits[id] /= self.repetition_penalty
            for sentence in self.history:
                for id in sentence:
                    next_token_logits[id] /= self.repetition_penalty

            next_token_logits = next_token_logits / self.temperature
            next_token_logits[self.tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=self.topk, top_p=self.topp)
            # torch.multinomial 根据权重无放回地抽取 num_samples 个元素, 返回元素下标
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            if next_token == self.tokenizer.sep_token_id:  # 遇到 [SEP] 表明生成结束
                break
            response.append(next_token.item())
            input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)
        self.history.append(response)
        self.history = self.history[-self.max_history_len:]
        response_tokens = self.tokenizer.convert_ids_to_tokens(response)
        return ("".join(response_tokens)).replace("#", "")


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocab size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def set_args():
    """
    Sets up the arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu', type=str, help='计算设备, 默认为 cpu')
    parser.add_argument('--model_dir', default='./eat_and_drink_full_bot_model/', type=str, help='模型存放目录')
    parser.add_argument('--topk', default=8, type=int, help='top_k_top_p_filtering 的参数 k, 生成时从概率最高的 k 个 token 中选择')
    parser.add_argument('--topp', default=0.0, type=float, help='top_k_top_p_filtering 的参数 p, 生成时从概率最高且累积概率不超过 p 的若干个 token 中选择')
    parser.add_argument('--temperature', default=1, type=float, help='生成时的 temperature. 值越高所生成文本的随机性越强, 默认值为 1')
    parser.add_argument('--repetition_penalty', default=1.5, type=float, help="重复惩罚参数. 若生成的对话重复性较高, 可适当提高该参数, 默认值为 1.5")
    parser.add_argument('--max_len', type=int, default=64, help='生成文本的最大长度. 超过指定长度则进行截断, 默认值为 64')
    parser.add_argument('--max_history_len', type=int, default=3, help="生成文本时考虑的前文对话轮数, 默认值为 3")
    return parser.parse_args()

def interact():
    args = set_args()
    inference = Inference(args.model_dir, args.device, args.max_history_len, args.max_len, 
                          args.repetition_penalty, args.temperature)
    print('开始和吃饱喝足 bot 聊天, 输入 break 或 end 结束')

    while True:
        try:
            query = input("user: ")
            if query.strip() in ('break', 'end'):
                raise ValueError("exit")
            text = inference.predict(query)
            print("bot: " + text)
        except (ValueError, EOFError):
            break


if __name__ == '__main__':
    interact()