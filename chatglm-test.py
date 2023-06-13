from transformers import AutoTokenizer, AutoModel
from model.xformer_chatglm import ChatGLMForConditionalGenerationXformer
from model.xformer_chatglm import change_config
from model.baseline_chatglm import ChatGLMForConditionalGeneration
from model.configuration_chatglm import ChatGLMConfig
import argparse
import yaml
import os
import torch


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text

def load_parameter(model_name: str, engine_use: bool):
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half()
    model = model.eval()

    configuration = ChatGLMConfig(
        bos_token_id=130004, 
        eos_token_id=130005, 
        mask_token_id=130000, 
        gmask_token_id=130001,
        pad_token_id=3,
        use_cache=True,
        vocab_size=130528,
        model_type="chatglm",
        torch_dtype="float16"
    )
    state_dict = model.state_dict()
    if engine_use:
        new_model = ChatGLMForConditionalGenerationXformer(configuration).eval()
        for i in range(configuration.num_layers):
            state_dict[f'transformer.layers.{i}.mlp.dense_h_to_4h_act.weight'] = state_dict.pop(f'transformer.layers.{i}.mlp.dense_h_to_4h.weight')
            state_dict[f'transformer.layers.{i}.mlp.dense_h_to_4h_act.bias'] = state_dict.pop(f'transformer.layers.{i}.mlp.dense_h_to_4h.bias')
    else:
        new_model = ChatGLMForConditionalGeneration(configuration).eval()

    new_model.load_state_dict(state_dict, strict=True)
    return new_model.half().cuda()

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq-len', type=int, default=-1)
    parser.add_argument('--test-case', type=int, default=-1)
    parser.add_argument('--engine-use', default=False, action='store_true')
    args = parser.parse_args()


    assert args.seq_len > -1 or args.test_case > -1, \
        "either seq len or test case should be assigned !"

    string = " "
    if args.seq_len == 8:
        file_name = os.path.join('./case', '8.yaml')
        case_id = 0
    elif args.seq_len == 16:
        file_name = os.path.join('./case', '16.yaml')
        case_id = 0
    elif args.seq_len == 128:
        file_name = os.path.join('./case', '128.yaml')
        case_id = 0
    elif args.seq_len == 256:
        file_name = os.path.join('./case', '256.yaml')
        case_id = 0
    elif args.seq_len == 512:
        file_name = os.path.join('./case', '512.yaml')
        case_id = args.test_case
    elif args.seq_len == 1024:
        file_name = os.path.join('./case', '1024.yaml')
        case_id = args.test_case
    else:
        dir = './case'
        file_list = os.listdir(dir)
        print(file_list)
        file_name = os.path.join('./case', file_list[args.test_case])
        case_id = 0
    f = open(file_name, 'r', encoding='utf-8')
    file = yaml.load(f, Loader=yaml.FullLoader)
    string = file[case_id]
    # string = "北京是一个怎样的城市"
    model_name = "THUDM/chatglm-6b"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = load_parameter(model_name, args.engine_use)
    model.transformer.duration = 0
    model.transformer.first_token_latency = 0
    model.transformer.forward_count = 0
    model.past_key_values = None
    # change_config(32, 64, 64)
    # torch.cuda.cudart().cudaProfilerStart()
    response, history = model.chat(tokenizer, parse_text(string), history=[])
    # torch.cuda.cudart().cudaProfilerStop()
    print("=======================================================")
    model.transformer.duration = 0
    model.transformer.first_token_latency = 0
    model.transformer.forward_count = 0
    model.past_key_values = None
    # torch.cuda.cudart().cudaProfilerStart()
    change_config(32, 64, 64)
    response, history = model.chat(tokenizer, parse_text(string), history=[])
    # torch.cuda.cudart().cudaProfilerStop()
    
    print(response)