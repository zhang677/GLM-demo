from transformers import AutoTokenizer, AutoModel
from model.modeling_chatglm import ChatGLMForConditionalGeneration
from model.configuration_chatglm import ChatGLMConfig
import argparse


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq-len', type=int, default=512)
    parser.add_argument('--engine-use', default=False, action='store_true')
    args = parser.parse_args()

string = " "
if args.seq_len == 8:
    string = "请回答北京是一个什么样的城市"
elif args.seq_len == 16:
    string = "北京曾举办过一次夏季奥运会。请回答北京是一个什么样的城市。"
elif args.seq_len == 128:
    string = "请你花时间思考你的未来，并设定明确的目标，建立一个具体的、可行的、可衡量的目标，包括长期和短期目标。\
        这些目标应该基于你的价值观和优势，并且是可以量化和具体化的。将目标分解为可操作的步骤，并将其加入到日程表中。\
        通过不断学习和提高自己的技能，为实现目标做好准备。选择一些合适的书籍、课程和社交平台等方式，不断提高自己的专业技能。\
        制定实际可行的计划，要全力以赴地去追求实现目标的机会和方式。不断探索和尝试新的机会和方式，不能够害怕失败和挫折，\
        而是积极地寻求帮助和支持，照顾好自己，保持健康。请总结这段话。"
elif args.seq_len == 256:
    string = "成都是中国西南地区的一个大都市，也是四川省的省会城市。\
        它拥有悠久的历史和文化传统，是中国西南地区最重要的经济、文化和交通中心之一。\
        成都是中国西南地区的一个著名城市，也是中国的四大古都之一。这个城市的历史可以追溯到公元前4000年左右的新石器时代。\
        成都地理位置优越，气候宜人，四季分明，温和湿润，被誉为“天府之国”的代表。 \
        成都的美食文化非常丰富多样，著名的川菜、火锅和串串香等美食享誉全球。\
        成都还是中国茶文化的十分重要发源地之一，四川茶也是中国十大名茶之一。\
        成都还拥有众多的历史和文化景点，包括武侯祠、锦里古街、杜甫草堂、青羊宫等等，吸引着大量的游客前来游览和探访。\
        成都经济发展迅速，是中国西部地区的经济中心之一，也是中国重要的高新技术产业基地。\
        成都拥有大量的高校和科研机构，培养了众多的优秀人才，推动了城市的发展和进步。\
        成都是一个充满活力和文化底蕴的城市，有着非常丰富的美食、景点和机会。无论您是来旅游、生活还是创业，成都都是一个值得一去的好地方。\
        成都是中国西南地区的一个历史悠久、文化底蕴深厚的城市，拥有丰富的地下资源和发达的交通系统，\
        同时也是熊猫之乡、商业中心和电子信息产业基地，生活丰富多彩，人民热情友好、开放包容。请总结这些对成都的描述。"
elif args.seq_len == 512:
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
elif args.seq_len == 1024:
    string = "人类交通的发展可以追溯到古代，从最早的步行和动物运输开始。然而，随着技术的进步和全球化的推动，交通系统得以显著改进和扩展。铁路交通：19世纪的工业革命催生了铁路的发展。铁路系统的建立连接了城市和国家，促进了商业和人员流动。\
        蒸汽机车的发明为铁路交通带来了重大突破，后来的电力化和高速铁路进一步提升了交通效率和速度。汽车交通：20世纪初，汽车的发明引领了个人交通的革命。大规模的汽车生产和道路基础设施建设使得汽车成为人们日常出行的主要工具。\
        随着时间的推移，汽车的设计和性能不断改进，包括引入燃油效率更高的发动机和安全性能提升的创新。航空交通：20世纪初的飞机发明开创了航空交通的新纪元。航空技术的进步使得飞机能够更快、更远地飞行，带来了全球航空网络的形成。\
        民航业的发展使得长途旅行变得更加便捷，同时也推动了国际贸易和旅游业的增长。船舶交通：海洋运输一直是国际贸易的重要组成部分。从古代的帆船到现代的大型货船和油轮，船舶技术的不断发展使得海上运输更加高效和可靠。\
        公共交通：随着城市化的加速和交通拥堵问题的出现，公共交通系统得到了广泛的关注和发展。\
        地铁、轻轨、公交车和电车等公共交通工具的建设和改进，为城市居民提供了便捷、环保的出行选择。同时，共享交通和出租车服务的兴起也为人们提供了灵活的选择。\
        汽车的发展是人类交通史上的重大里程碑之一。从最早的蒸汽驱动汽车到现代电动汽车的兴起，汽车技术经历了巨大的变革和进步。以下是关于汽车发展的一些主要方面：\
        起源和早期发展：汽车的起源可以追溯到18世纪末和19世纪初。早期的汽车多采用蒸汽引擎，其中最著名的是尼古拉斯·约瑟夫·庞巴迪的蒸汽马车。\
        然而，随着内燃机的发明，汽油和柴油驱动的汽车逐渐取代了蒸汽驱动的汽车。大规模生产和亨利·福特：20世纪初，亨利·福特引领了汽车产业的革命。他的创新包括流水线生产和大规模制造模式，使得汽车生产变得高效且成本降低。\
        福特于1908年推出了著名的“T型车”，使汽车普及化，为大众交通带来了革命性的变化。设计和性能改进：汽车制造商在设计和性能方面进行了持续的改进。引入了气动外形设计、更高效的发动机和先进的底盘技术，提高了汽车的燃油经济性、安全性和驾驶体验。\
        创新的材料和制造工艺使得汽车更轻便、坚固和环保。飞机的发展自人类向天空展翅飞翔的梦想诞生以来，已经经历了令人瞩目的进步和变革。从最早的飞行器原型到今天的超音速喷气式客机和先进的无人机技术，飞机已经成为人类生活和全球经济的重要组成部分。\
        飞机的历史可以追溯到公元前5世纪，古希腊哲学家阿基米德设计了一种被称为“阿基米德螺旋”或“空气螺旋”的飞行器原型。然而，真正的飞机发展始于18世纪末和19世纪初，当时许多先驱者开始研究和试验各种飞行原理。\
        其中最为著名的是莱特兄弟，他们于1903年成功飞行了世界上第一架受人操纵的飞机，标志着现代航空的开始。自那时以来，飞机的发展取得了巨大的进步。早期的飞机主要采用螺旋桨作为动力源，而在第一次世界大战期间，飞机被广泛运用于军事行动。到了20世纪30年代，喷气式发动机的发明引领了飞机技术的新时代。\
        1949年，英国的“喷气式飞跃”实现了世界上第一次喷气式飞机的商业航班，标志着喷气时代的来临。在接下来的几十年里，飞机的设计和技术不断突破。喷气式客机的速度和载客能力大大提高，航空业迅速发展。随着航空工程的进步，涡轮螺旋桨飞机、超音速飞机和宽体客机相继问世。\
        20世纪60年代，喷气式客机的规模进一步扩大，波音747问世，成为当时最大的客机。此后，空中客车公司也推出了自己的宽体客机系列，如空中客车A380。\
        自人类掌握航海技术以来，船舶一直是人类探索和贸易的重要工具。从最早的木制划船到如今的现代化巨轮，轮船的发展经历了漫长而精彩的历程。\
        古代文明如古埃及、古希腊和古罗马都有自己的船舶建造技术和航行知识。然而，真正的轮船发展开始于19世纪。\
        蒸汽船的出现被视为轮船发展的重要里程碑。早期的蒸汽船使用蒸汽机驱动船轮，取代了传统的风帆和桨。第一艘商业化的蒸汽船是由美国工程师罗伯特·弗尔顿于1807年设计和建造的克莱蒙特号，它在哈德逊河上进行了首次试航。\
        这一创举引发了全球范围内对蒸汽船的兴趣，并催生了大量的蒸汽船公司和航线。随着工业革命的到来，蒸汽船的发展进一步加速。蒸汽机的改进和铁质船体的使用使得轮船的速度、负载能力和舒适度大大提高。\
        在19世纪中叶和下半叶，大西洋横渡成为蒸汽船的主要航线之一，远洋航行变得更加便捷和可靠。然而，随着内燃机技术的进步和石油的广泛应用，蒸汽船逐渐被内燃机驱动的船舶取代。内燃机船舶采用柴油发动机作为动力源，具有更高的效率和更低的运营成本。\
        20世纪初，柴油船成为主导船舶市场的力量。请回答从这段世界交通发展的描述中得到的启发。"
else:
    raise NotImplementedError

model_name = "THUDM/chatglm-6b"

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
        engine_use=args.engine_use,
        tiny=tiny_bool
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
        if tiny_bool:
            new_model.transformer.layers[i].attention_query_key_value_weight = model.transformer.layers[i].attention.query_key_value.weight.transpose(0, 1).contiguous()
            new_model.transformer.layers[i].attention_dense_weight = model.transformer.layers[i].attention.dense.weight.transpose(0, 1).contiguous()
            new_model.transformer.layers[i].dense_h_to_4h_weight = model.transformer.layers[i].mlp.dense_h_to_4h.weight.transpose(0, 1).contiguous()
            new_model.transformer.layers[i].dense_4h_to_h_weight = model.transformer.layers[i].mlp.dense_4h_to_h.weight.transpose(0, 1).contiguous()
    new_model.transformer.final_layernorm.weight = model.transformer.final_layernorm.weight
    new_model.transformer.final_layernorm.bias = model.transformer.final_layernorm.bias
    new_model.half().cuda()
    new_model = new_model.eval()

    return new_model

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = load_parameter(model_name, args.seq_len)
response, history = model.chat(tokenizer, string, history=[])
print(response)