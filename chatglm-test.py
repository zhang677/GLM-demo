from transformers import AutoTokenizer, AutoModel
from model.modeling_chatglm import ChatGLMForConditionalGeneration
from model.configuration_chatglm import ChatGLMConfig

#print(model.__class__)
# [8]
# string = "请回答北京是一个什么样的城市"
# [16]
# string = "北京曾举办过一次夏季奥运会。请回答北京是一个什么样的城市。"
# [128]
# string = "请你花时间思考你的未来，并设定明确的目标，建立一个具体的、可行的、可衡量的目标，包括长期和短期目标。这些目标应该基于你的价值观和优势，并且是可以量化和具体化的。将目标分解为可操作的步骤，并将其加入到日程表中。通过不断学习和提高自己的技能，为实现目标做好准备。选择一些合适的书籍、课程和社交平台等方式，不断学习和提高自己的专业技能和社交能力。制定实际可行的计划，要全力以赴地去追求实现目标的机会和方式。不断探索和尝试新的机会和方式，不能够害怕失败和挫折，而是积极地寻求帮助和支持，照顾好自己，保持健康。请总结这段话。"
# [256]
# string = "成都是中国西南地区的一个大都市，也是四川省的省会城市。\
#     它拥有悠久的历史和文化传统，是中国西南地区最重要的经济、文化和交通中心之一。\
#     成都是中国西南地区的一个著名城市，也是中国的四大古都之一。这个城市的历史可以追溯到公元前4000年左右的新石器时代。\
#     成都地理位置优越，气候宜人，四季分明，温和湿润，被誉为“天府之国”的代表。 \
#     成都的美食文化非常丰富多样，著名的川菜、火锅和串串香等美食享誉全球。\
#     成都还是中国茶文化的十分重要发源地之一，四川茶也是中国十大名茶之一。\
#     成都还拥有众多的历史和文化景点，包括武侯祠、锦里古街、杜甫草堂、青羊宫等等，吸引着大量的游客前来游览和探访。\
#     成都经济发展迅速，是中国西部地区的经济中心之一，也是中国重要的高新技术产业基地。\
#     成都拥有大量的高校和科研机构，培养了众多的优秀人才，推动了城市的发展和进步。\
#     成都是一个充满活力和文化底蕴的城市，有着非常丰富的美食、景点和机会。无论您是来旅游、生活还是创业，成都都是一个值得一去的好地方。\
#     成都是中国西南地区的一个历史悠久、文化底蕴深厚的城市，拥有丰富的地下资源和发达的公共交通系统，同时也是熊猫之乡、商业中心和电子信息产业基地，夜生活丰富多彩，人民热情友好、开放包容。请总结这些对成都的描述。"
# [512]
string = "网球是一项源远流长的体育运动，其历史可以追溯到数百年前。据考古学家的研究，类似于网球的活动在古代的埃及、希腊、罗马等地就已经存在了，但是那时的球场都是非常简陋的，没有像现在这样的专业设施和比赛规则。\
    古代的球类运动在不同文明中都有出现，比如古埃及的“沙托克”和古希腊的“斯芬克斯”，但是这些运动和现代网球之间存在一定的差异。真正的网球运动起源于法国，又称“皇家网球”，源于16世纪的宫廷活动，后来逐渐传播到了英国。\
    19世纪初，英国的贵族们开始在室内场地上打网球。\
    这种运动需要特制的球场，球场内有四堵墙，其中一堵有一条网隔开，球员需要用木质球拍把羊毛球打到另一方的墙上，使其落在对方无法接到的区域，得分赢得比赛。\
    19世纪中期，网球开始演变成为现代网球，主要原因是人们开始在室外场地上进行比赛，同时也改进了球拍和球的质量。1877年，第一届温布尔登网球公开赛举行，这也是世界上最古老、最著名的网球赛事之一，至今已经成为了网球世界的重要盛事。 \
    自20世纪以来，网球逐渐成为了一项全球性的运动，除了温布尔登之外，还有法国网球公开赛、美国网球公开赛和澳大利亚网球公开赛等重要的大满贯赛事。\
    20世纪后期，网球运动迅速发展，全球范围内涌现出越来越多的网球选手和比赛。现在，网球已经成为一项全球性的运动和娱乐活动，其历史和文化价值也被越来越多的人们所重视。 \
    介绍一位网球传奇，罗杰·费德勒，费德勒出生于1981年，自小就表现出了对网球的天赋。在他十几岁的时候，已经开始在国际比赛中崭露头角。2003年，费德勒在温布尔登赛上获得了自己的第一个大满贯单打冠军，从此开始了他在网球界的辉煌历程。\
    在费德勒的职业生涯中，他赢得了20个大满贯单打冠军，是历史上单打大满贯冠军数最多的男子选手。\
    除了他的荣誉和成就，费德勒还因其独特的球技和优雅的打法而备受推崇。他经常运用一些高难度的技巧，如穿越式过网球、单手反拍、网前拍击等等，这些技巧在网球比赛中很少出现，但是费德勒却能够运用自如，让他成为了一个非常具有观赏性的选手。 \
    他的成就和影响不仅仅局限于网球界，更是让人们看到了一个追求卓越和不懈努力的典范，这也是网球的价值所在。\
    总的来说，网球经历了数百年的发展，成为了一项全球性的运动和娱乐活动，其历史和文化背景也与人类社会的发展和变迁密切相关。你认为未来网球将会如何发展？"
# response, history = model.chat(tokenizer, string, history=[])
# print(response)
#for (param_id, p) in enumerate(model.transformer.layers[0].parameters()):
#    if p.requires_grad:
#        print(param_id, p.shape)
        #model.transformer.layers[1].parameters()[param_id] = p.half().cuda()
# transformer_layers = model.transformer.layers
#print(transformer_layers[1].__class__)

model_name = "THUDM/chatglm-6b"

def load_parameter(model_name: str):
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda()
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
        torch_dtype="float16",
        # switch on the accelerating engine
        engine_use=True
    )

    # model = model.cpu()
    new_model = ChatGLMForConditionalGeneration(configuration)
    #print(new_model.transformer.layers[1].mlp.dense_4h_to_h.bias)
        #print(model.transformer.layers[0].mlp.dense_4h_to_h.bias)
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
        # new_model.transformer.layers[i].attention_query_key_value_weight = model.transformer.layers[i].attention.query_key_value.weight.transpose(0, 1).contiguous()
        # new_model.transformer.layers[i].attention_dense_weight = model.transformer.layers[i].attention.dense.weight.transpose(0, 1).contiguous()
        # new_model.transformer.layers[i].dense_h_to_4h_weight = model.transformer.layers[i].mlp.dense_h_to_4h.weight.transpose(0, 1).contiguous()
        # new_model.transformer.layers[i].dense_4h_to_h_weight = model.transformer.layers[i].mlp.dense_4h_to_h.weight.transpose(0, 1).contiguous()
    new_model.transformer.final_layernorm.weight = model.transformer.final_layernorm.weight
    new_model.transformer.final_layernorm.bias = model.transformer.final_layernorm.bias
    new_model.half().cuda()
    new_model = new_model.eval()

    return new_model

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = load_parameter(model_name)
response, history = model.chat(tokenizer, string, history=[])
print(response)