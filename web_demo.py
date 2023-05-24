from transformers import AutoModel, AutoTokenizer
import gradio as gr
import mdtex2html
import torch
import os
import yaml
import time
# from threading import Thread
from model.xformer_chatglm import ChatGLMForConditionalGenerationXformer
from model.baseline_chatglm import ChatGLMForConditionalGeneration
from model.configuration_chatglm import ChatGLMConfig

model_name = "THUDM/chatglm-6b"
torch.random.manual_seed(999)

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


tokenizer_1 = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model_1 = load_parameter(model_name, False)
model_1 = model_1.eval()
model_1.half().to("cuda:1")

tokenizer_2 = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model_2 = load_parameter(model_name, True)
model_2 = model_2.eval()
model_2.half().to("cuda:0")

print(model_1.device)
print(model_2.device)
# model_2.cuda(1)


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


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

def predict_1(input, chatbot, history):
    chatbot.append((parse_text(input), ""))
    # for response, history in model_1.stream_chat(tokenizer_1, input, history, max_length=max_length, top_p=top_p,
    #                                            temperature=temperature):
    #     chatbot[-1] = (parse_text(input), parse_text(response))       

    #     yield chatbot, history
    model_1.transformer.duration = 0
    model_1.transformer.first_token_latency = 0
    model_1.transformer.forward_count = 0
    model_1.past_key_values = None
    # warm up
    response, history = model_1.chat(tokenizer_1, input, history=[])
    for response, history in model_1.stream_chat(tokenizer_1, input, history=[]):
        chatbot[-1] = (parse_text(input), parse_text(response))   
        yield chatbot, history, "……", "……"
    end_to_end = model_1.transformer.duration
    first_token = model_1.transformer.first_token_latency
    yield chatbot, history, end_to_end, first_token
    return


def predict_2(input, chatbot, history):
    chatbot.append((parse_text(input), ""))
    # for response, history in model_2.stream_chat(tokenizer_2, input, history, max_length=max_length, top_p=top_p,
    #                                            temperature=temperature):
    #     chatbot[-1] = (parse_text(input), parse_text(response))       

    #     yield chatbot, history
    model_2.transformer.duration = 0
    model_2.transformer.first_token_latency = 0
    model_2.transformer.forward_count = 0
    model_2.past_key_values = None
    # warm up
    response, history = model_2.chat(tokenizer_2, input, history=[])
    for response, history in model_2.stream_chat(tokenizer_2, input, history=[]):
        chatbot[-1] = (parse_text(input), parse_text(response))   
        yield chatbot, history, "……", "……"
    end_to_end = model_2.transformer.duration
    first_token = model_2.transformer.first_token_latency
    yield chatbot, history, end_to_end, first_token
    return 


def autotest_1(chatbot, history):
    chatbot = []
    history = []
    yield chatbot, history, "……", "……"
    e_list = []
    f_list = []
    dir = './test_case'
    file_list = os.listdir(dir)
    for test_i in range(7):
        time.sleep(2.0)
        file_name = os.path.join(dir, file_list[test_i])
        f = open(file_name, 'r', encoding='utf-8')
        file = yaml.load(f, Loader=yaml.FullLoader)
        input = file[0]
        chatbot.append((parse_text(input), ""))
        model_1.transformer.duration = 0
        model_1.transformer.first_token_latency = 0
        model_1.transformer.forward_count = 0
        model_1.past_key_values = None
        # warm up
        response, history = model_1.chat(tokenizer_1, input, history=[])
        for response, history in model_1.stream_chat(tokenizer_1, input, history=[]):
            chatbot[-1] = (parse_text(input), parse_text(response))  
            yield chatbot, history, "……", "……" 
        history = []
        end_to_end = model_1.transformer.duration
        first_token = model_1.transformer.first_token_latency
        e_list.append(end_to_end)
        f_list.append(first_token)
        yield chatbot, history, end_to_end, first_token
    e_mean = sum(e_list) / len(e_list)
    f_mean = sum(f_list) / len(f_list)
    yield chatbot, history, e_mean, f_mean
    return 


def autotest_2(chatbot, history):
    chatbot = []
    history = []
    yield chatbot, history, "……", "……"
    e_list = []
    f_list = []
    dir = './test_case'
    file_list = os.listdir(dir)
    for test_i in range(7):
        time.sleep(2.0)
        file_name = os.path.join(dir, file_list[test_i])
        f = open(file_name, 'r', encoding='utf-8')
        file = yaml.load(f, Loader=yaml.FullLoader)
        input = file[0]
        chatbot.append((parse_text(input), ""))
        model_2.transformer.duration = 0
        model_2.transformer.first_token_latency = 0
        model_2.transformer.forward_count = 0
        model_2.past_key_values = None
        # warm up
        response, history = model_2.chat(tokenizer_2, input, history=[])
        for response, history in model_2.stream_chat(tokenizer_2, input, history=[]):
            chatbot[-1] = (parse_text(input), parse_text(response))   
            yield chatbot, history, "……", "……"
        history = []
        end_to_end = model_2.transformer.duration
        first_token = model_2.transformer.first_token_latency
        e_list.append(end_to_end)
        f_list.append(first_token)
        yield chatbot, history, end_to_end, first_token
    e_mean = sum(e_list) / len(e_list)
    f_mean = sum(f_list) / len(f_list)
    yield chatbot, history, e_mean, f_mean
    return 


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], [], [], []


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">ChatGLM Toy</h1>""")

    with gr.Column():
        with gr.Row(scale=3):
            with gr.Column(scale=1): 
                with gr.Column(scale=3):
                    chatbot_1 = gr.Chatbot()
                    time_1 = gr.Textbox(label="End-to-End Latency [ms]")
                    time_1f = gr.Textbox(label="First-Token Latency [ms]")
                # with gr.Column(scale=1):
                #     max_length_1 = gr.Slider(0, 4096, value=2048, step=1.0, label="Maximum length", interactive=True)
                #     top_p_1 = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
                #     temperature_1 = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)
            with gr.Column(scale=1): 
                with gr.Column(scale=3):   
                    chatbot_2 = gr.Chatbot()
                    time_2 = gr.Textbox(label="End-to-End Latency [ms]")
                    time_2f = gr.Textbox(label="First-Token Latency [ms]")
                # with gr.Column(scale=1):
                #     max_length_2 = gr.Slider(0, 4096, value=2048, step=1.0, label="Maximum length", interactive=True)
                #     top_p_2 = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
                #     temperature_2 = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)
        with gr.Row(scale=2):
            with gr.Column():
                with gr.Column(scale=12):
                    user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                        container=False)
                with gr.Column(min_width=32, scale=1):
                    submitBtn = gr.Button("Submit", variant="primary")
                    emptyBtn = gr.Button("Clear History")
                    testBtn = gr.Button("AutoTest")

    history_1 = gr.State([])
    history_2 = gr.State([])

    submitBtn.click(predict_2, [user_input, chatbot_2, history_2], \
                            [chatbot_2, history_2, time_2, time_2f],
                            show_progress=True)
    
    submitBtn.click(predict_1, [user_input, chatbot_1, history_1], \
                            [chatbot_1, history_1, time_1, time_1f],
                            show_progress=True)
    
    submitBtn.click(reset_user_input, [], [user_input])

    testBtn.click(autotest_1, [chatbot_1, history_1], \
                            [chatbot_1, history_1, time_1, time_1f],)
    testBtn.click(autotest_2, [chatbot_2, history_2], \
                            [chatbot_2, history_2, time_2, time_2f],)

    emptyBtn.click(reset_state, outputs=[chatbot_1, history_1, chatbot_2, history_2], show_progress=True)

# demo.queue().launch(share=False, inbrowser=True)
demo.queue(concurrency_count=3)
demo.launch(share=True, inbrowser=True)
