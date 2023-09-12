import os
import platform
import signal
from transformers import AutoTokenizer, AutoModel
import readline

import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.gpu_migration

import time

import argparse

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
model = model.eval()
model = ht.hpu.wrap_in_hpu_graph(model)

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False


def build_prompt(history):
    prompt = "欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM-6B：{response}"
    return prompt


def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True


def setup_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='ChatGLM2-6B for HPU')
    parser.add_argument('--prompt', '-p', type=int, help="Number of input prompt tokens", default=128)
    parser.add_argument('--repeat', '-r', type=int, help="Number of times each query should be repeated", default=5)
    parser.add_argument('--max_new_tokens', '-t', type=int, help="Number of maximum new tokens", default=128)
    parser.add_argument('--batch_size', '-b', type=int, help="Number of times each query should be repeated", default=4)
    return parser


def get_prompts(input_prompt=128, max_new_tokens=128, prompt_repeat=5):
    queries = list()
    if input_prompt == 128:
        prompt_128 = "患者男，年龄29岁，血型O，因思维迟钝，易激怒，因发热伴牙龈出血14天，乏力、头晕5天就诊我院急诊科。快速完善检查，血常规显示患者三系血细胞重度减低，凝血功能检查提示APTT明显延长，纤维蛋白原降低，血液科会诊后发现患者高热、牙龈持续出血，胸骨压痛阳性.于3903年3月7日入院治疗，出现头痛头晕。答："
        queries.append(prompt_128)
    elif input_prompt == 1024:
        prompt_1024 = "患者男，年龄29岁，血型O，因思维迟钝，易激怒，因发热伴牙龈出血14天，乏力、头晕5天就诊我院急诊科。快速完善检查，血常规显示患者三系血细胞重度减低，凝血功能检查提示APTT明显延长，纤维蛋白原降低，血液科会诊后发现患者高热、牙龈持续出血，胸骨压痛阳性.于3903年3月7日入院治疗，出现头痛、头晕、伴发热（最高体温42℃）症状，曾到其他医院就医。8日症状有所好转，9日仍有头痛、呕吐，四肢乏力伴发热。10日凌晨到本院就诊。患者5d前出现突发性思维迟钝，脾气暴躁，略有不顺心就出现攻击行为，在院外未行任何诊治。既往身体健康，平素性格内向。体格检查无异常。血常规白细胞中单核细胞百分比升高。D-二聚体定量1412μg/L，骨髓穿刺示增生极度活跃，异常早幼粒细胞占94%.外周血涂片见大量早幼粒细胞，并可在胞浆见到柴捆样细胞.以下是血常规详细信息：1.病人红细胞计数结果：3.2 x10^12/L. 附正常参考范围：新生儿:（6.0～7.0）×10^12/L；婴儿：（5.2～7.0）×10^12/L; 儿童：（4.2～5.2）×10^12/L; 成人男：（4.0～5.5）×10^12/L; 成人女：（3.5～5.0）×10^12/L. 临床意义：生理性红细胞和血红蛋白增多的原因：精神因素（冲动、兴奋、恐惧、冷水浴刺激等导致肾上腺素分泌增多的因素）、红细胞代偿性增生（长期低气压、缺氧刺激，多次献血）；生理性红细胞和血红蛋白减少的原因：造血原料相对不足，多见于妊娠、6个月～2岁婴幼儿、某些老年性造血功能减退；病理性增多：多见于频繁呕吐、出汗过多、大面积烧伤、血液浓缩，慢性肺心病、肺气肿、高原病、肿瘤以及真性红细胞增多症等；病理性减少：多见于白血病等血液系统疾病；急性大出血、严重的组织损伤及血细胞的破坏等；合成障碍，见于缺铁、维生素B12缺乏等。2. 病人血红蛋白测量结果：108g/L. 附血红蛋白正常参考范围：男性120～160g/L；女性110～150g/L；新生儿170～200g/L；临床意义：临床意义与红细胞计数相仿，但能更好地反映贫血程度，极重度贫血（Hb<30g/L）、重度贫血（31～60g/L）、中度贫血（61～90g/L）、男性轻度贫血（90~120g/L）、女性轻度贫血（90~110g/L）。3. 病人白细胞计数结果：13.6 x 10^9/L; 附白细胞计数正常参考范围：成人（4.0～10.0）×10^9/L；新生儿（11.0～12.0）×10^9/L。临床意义：1）生理性白细胞计数增高见于剧烈运动、妊娠、新生儿；2）病理性白细胞增高见于急性化脓性感染、尿毒症、白血病、组织损伤、急性出血等；3）病理性白细胞减少见于再生障碍性贫血、某些传染病、肝硬化、脾功能亢进、放疗化疗等。4. 病人白细胞分类技术结果：中性粒细胞（N）50%、嗜酸性粒细胞（E）3.8%、嗜碱性粒细胞（B）0.2%、淋巴细胞（L）45%、单核细胞（M）1%。附白细胞分类计数正常参考范围：中性粒细胞（N）50%～70%。问：基于以上信息做出判断，该患者是否有罹患急性白血病的风险？请结合上述内容给出判断的详细解释，并简要总结潜在的早期征兆、预防方法、相关的基因突变、常用的治疗手段，以及当前已上市和正在临床阶段的药物清单。答："
        queries.append(prompt_1024)
    else:
        assert False, "input prompt length {} not supported!".format(input_prompt)
    queries.append("clear")
    queries = queries * prompt_repeat
    max_length = input_prompt + max_new_tokens
    return queries, max_length


def main():
    parser = setup_parser()
    args = parser.parse_args()
    queries, max_length = get_prompts(args.prompt, args.max_new_tokens, args.repeat)

    print("[batch_size] {} [run prompt] {} [run repeat] {} [max_new_tokens] {}".format(
        args.batch_size, args.prompt, args.repeat, args.max_new_tokens))

    history = []
    global stop_stream
    print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    for i, query in enumerate(queries):
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            os.system(clear_command)
            print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        count = 0
        for response, history in model.stream_chat(tokenizer, query, history=history,
                                                   max_length=max_length, bs_repeat=args.batch_size):
            if stop_stream:
                stop_stream = False
                break
            else:
                count += 1
                if count % 8 == 0:
                    os.system(clear_command)
                    print(build_prompt(history), flush=True)
                    signal.signal(signal.SIGINT, signal_handler)
        os.system(clear_command)
        print(build_prompt(history), flush=True)


if __name__ == "__main__":
    main()
