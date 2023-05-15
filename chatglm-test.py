from transformers import AutoTokenizer, AutoModel
from model.modeling_chatglm import ChatGLMForConditionalGenerationByte
from model.baseline_chatglm import ChatGLMForConditionalGeneration
from model.configuration_chatglm import ChatGLMConfig
import argparse
import yaml
import os
import torch

def load_parameter(model_name: str, seq_len: int):
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda()
    model = model.eval()

    tiny_bool = (seq_len <= 256)

    configuration = ChatGLMConfig(
        bos_token_id=130004, 
        eos_token_id=130005, 
        mask_token_id=130000, 
        gmask_token_id=130001,
        pad_token_id=3,
        use_cache=True,
        vocab_size=130528,
        model_type="chatglm",
        torch_dtype="float16",
        # switch on the accelerating engine
        # engine_use=args.engine_use,
        # tiny=tiny_bool
    )

    # model = model.cpu()
    if args.engine_use:
        configuration.engine_use = args.engine_use
        configuration.tiny = tiny_bool
        new_model = ChatGLMForConditionalGenerationByte(configuration)
    else:
        new_model = ChatGLMForConditionalGeneration(configuration)
    
    new_model.transformer.word_embeddings.weight = model.transformer.word_embeddings.weight
    new_model.lm_head.weight = model.lm_head.weight
    for i in range(configuration.num_layers):
        new_model.transformer.layers[i].input_layernorm.weight = model.transformer.layers[i].input_layernorm.weight
        new_model.transformer.layers[i].input_layernorm.bias = model.transformer.layers[i].input_layernorm.bias
        new_model.transformer.layers[i].attention.query_key_value.weight = model.transformer.layers[i].attention.query_key_value.weight
        new_model.transformer.layers[i].attention.query_key_value.bias = model.transformer.layers[i].attention.query_key_value.bias
        new_model.transformer.layers[i].attention.dense.weight = model.transformer.layers[i].attention.dense.weight
        new_model.transformer.layers[i].attention.dense.bias = model.transformer.layers[i].attention.dense.bias
        new_model.transformer.layers[i].post_attention_layernorm.weight = model.transformer.layers[i].post_attention_layernorm.weight
        new_model.transformer.layers[i].post_attention_layernorm.bias = model.transformer.layers[i].post_attention_layernorm.bias
        new_model.transformer.layers[i].mlp.dense_h_to_4h.weight = model.transformer.layers[i].mlp.dense_h_to_4h.weight
        new_model.transformer.layers[i].mlp.dense_h_to_4h.bias = model.transformer.layers[i].mlp.dense_h_to_4h.bias
        new_model.transformer.layers[i].mlp.dense_4h_to_h.weight = model.transformer.layers[i].mlp.dense_4h_to_h.weight
        new_model.transformer.layers[i].mlp.dense_4h_to_h.bias = model.transformer.layers[i].mlp.dense_4h_to_h.bias
        # load parameters into byte transformer backend
        if tiny_bool and args.engine_use:
            new_model.transformer.layers[i].attention_query_key_value_weight = model.transformer.layers[i].attention.query_key_value.weight.transpose(0, 1).contiguous()
            new_model.transformer.layers[i].attention_dense_weight = model.transformer.layers[i].attention.dense.weight.transpose(0, 1).contiguous()
            new_model.transformer.layers[i].dense_h_to_4h_weight = model.transformer.layers[i].mlp.dense_h_to_4h.weight.transpose(0, 1).contiguous()
            new_model.transformer.layers[i].dense_4h_to_h_weight = model.transformer.layers[i].mlp.dense_4h_to_h.weight.transpose(0, 1).contiguous()
    new_model.transformer.final_layernorm.weight = model.transformer.final_layernorm.weight
    new_model.transformer.final_layernorm.bias = model.transformer.final_layernorm.bias
    new_model.half().cuda()
    new_model = new_model.eval()

    return new_model

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq-len', type=int, default=-1)
    parser.add_argument('--test-case', type=int, default=-1)
    parser.add_argument('--engine-use', default=False, action='store_true')
    args = parser.parse_args()

    torch.ops.load_library('./lib/libths_bytetransformer.so')

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
    f = open(file_name, 'r')
    file = yaml.load(f, Loader=yaml.FullLoader)
    string = file[case_id]

    model_name = "THUDM/chatglm-6b"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = load_parameter(model_name, args.seq_len)
    response, history = model.chat(tokenizer, string, history=[])
    print(response)