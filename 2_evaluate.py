import re
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from pypinyin import lazy_pinyin, Style

#全局配置
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
SPECIAL_TOKENS = {"<PAD>", "<SOP>", "<EOP>", "<UNK>", "<PLACEHOLDER>"}
PUNC_REG = re.compile(r"[，。]")
MAX_RETRY = 3
QUALIFY_THRESHOLD = {
    "format_min": 0.8,
    "repeat_max": 0.2,
}
#7组固定测试题目（顺序对应txt每行）
test_set = [
    {"mode": "prefix_continue", "genre": "wuyan_jueju", "prompt": "丽日照残春"},
    {"mode": "cangtou", "genre": "qiyan_jueju", "prompt": "深度学习"},
    {"mode": "normal", "genre": "qiyan_jueju", "prompt": "月夜山居"},
    {"mode": "normal", "genre": "qiyan_jueju", "prompt": "江畔黄昏"},
    {"mode": "normal", "genre": "qiyan_jueju", "prompt": "江楼晚忆"},
    {"mode": "normal", "genre": "qiyan_jueju", "prompt": "秋日闲思"},
    {"mode": "normal", "genre": "qiyan_jueju", "prompt": "都市记忆"},
]
prompt_names = [item["prompt"] for item in test_set]

#自动评估核心类别
class PoemEvaluator:
    def clean(self, text):
        text = text.replace("\n", "").replace(" ", "")
        sentences = PUNC_REG.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        pure_char = PUNC_REG.sub("", text)
        return sentences, pure_char

    def token_leak(self, text):
        leak_count = sum(text.count(tok) for tok in SPECIAL_TOKENS)
        return {"token_leak_num": leak_count}

    def format_score(self, sentences, genre):
        if genre == "wuyan_jueju":
            std_word, std_sent = 5, 4
        elif genre == "qiyan_jueju":
            std_word, std_sent = 7, 4
        else:
            return {"format_score": 1.0, "sent_count": len(sentences)}
        sent_standard = 1 if len(sentences) == std_sent else 0
        word_standard = sum(1 for s in sentences if len(s) == std_word) / std_sent
        score = round((sent_standard + word_standard) / 2, 4)
        return {"format_score": score, "sent_count": len(sentences)}

    def repeat_rate(self, pure_char, sentences):
        if len(pure_char) == 0:
            return {"char_repeat": 0.0, "bigram_repeat": 0.0, "unique_word_rate": 0.0}
        char_counter = Counter(pure_char)
        char_repeat = sum(v - 1 for v in char_counter.values() if v > 1) / len(pure_char)
        bigrams = []
        for s in sentences:
            bigrams.extend([s[i:i+2] for i in range(len(s)-1)])
        bg_repeat = 0.0
        if bigrams:
            bg_counter = Counter(bigrams)
            bg_repeat = sum(v - 1 for v in bg_counter.values() if v > 1) / len(bigrams)
        unique_rate = round(len(char_counter) / len(pure_char), 4)
        return {
            "char_repeat": round(char_repeat, 4),
            "bigram_repeat": round(bg_repeat, 4),
            "unique_word_rate": unique_rate
        }

    def cangtou_check(self, sentences, head_str):
        if len(sentences) < 4 or len(head_str) < 4:
            return {"cangtou_acc": 0.0}
        hit = sum(1 for i in range(4) if sentences[i][0] == head_str[i])
        return {"cangtou_acc": round(hit / 4, 4)}

    def get_rhyme(self, v_char):
        """提取汉字韵母，用于押韵判断"""
        pys = lazy_pinyin(v_char, style=Style.FINALS)
        if not pys:
            return ""
        yun = pys[0]
        #韵尾简化（同韵类归为一组，通用简易押韵规则）
        yun_map = {
            "i": "i", "u": "u", "v": "u",
            "a": "a", "ia": "a", "ua": "a",
            "o": "o", "uo": "o", "e": "e", "ie": "e", "ve": "e",
            "ai": "ai", "uai": "ai", "ei": "ei", "ui": "ei",
            "ao": "ao", "iao": "ao", "ou": "ou", "iu": "ou",
            "an": "an", "ian": "an", "uan": "an", "van": "an",
            "en": "en", "in": "en", "un": "en", "vn": "en",
            "ang": "ang", "iang": "ang", "uang": "ang",
            "eng": "eng", "ing": "eng", "ong": "eng", "iong": "eng"
        }
        return yun_map.get(yun, yun)

    def rhyme_check(self, sentences, genre):
        if genre not in ("wuyan_jueju", "qiyan_jueju") or len(sentences) < 4:
            return {"rhyme_ok": 0, "rhyme_2": "", "rhyme_4": ""}
        char2 = sentences[1][-1]
        char4 = sentences[3][-1]
        yun2 = self.get_rhyme(char2)
        yun4 = self.get_rhyme(char4)
        rhyme_ok = 1 if yun2 == yun4 and yun2 != "" else 0
        return {"rhyme_ok": rhyme_ok, "rhyme_2": yun2, "rhyme_4": yun4}

    def single_eval(self, text, meta):
        sen_list, pure_text = self.clean(text)
        res = {}
        res.update(self.token_leak(text))
        res.update(self.format_score(sen_list, meta["genre"]))
        res.update(self.repeat_rate(pure_text, sen_list))
        #统一填充藏头准确率，无藏头题目填0避免绘图报错
        if meta["mode"] == "cangtou":
            res.update(self.cangtou_check(sen_list, meta["prompt"]))
        else:
            res["cangtou_acc"] = 0.0
        res.update(self.rhyme_check(sen_list, meta["genre"]))
        res["prompt"] = meta["prompt"]
        res["genre"] = meta["genre"]
        res["mode"] = meta["mode"]
        res["raw_poem"] = text
        #综合总分计算
        total = res["format_score"] * 0.3 + (1 - res["bigram_repeat"]) * 0.3 + res["unique_word_rate"] * 0.2
        if res["rhyme_ok"] == 1:
            total += 0.15
        if meta["mode"] == "cangtou":
            total += res["cangtou_acc"] * 0.05
        res["total_score"] = round(total, 4)
        return res

    def is_qualified(self, eval_result, meta):
        if eval_result["format_score"] < QUALIFY_THRESHOLD["format_min"]:
            return False, f"格式分{eval_result['format_score']}不足阈值"
        if eval_result["token_leak_num"] > 0:
            return False, f"存在特殊标记泄露{eval_result['token_leak_num']}个"
        if eval_result["bigram_repeat"] > QUALIFY_THRESHOLD["repeat_max"]:
            return False, f"短语重复率{eval_result['bigram_repeat']}过高"
        if meta["mode"] == "cangtou" and eval_result["cangtou_acc"] < 1.0:
            return False, f"藏头匹配不全{eval_result['cangtou_acc']}"
        if eval_result["rhyme_ok"] == 0:
            return False, "二四句不押韵"
        return True, "合格"

    def batch_run(self, meta_list, poem_list, save_file):
        all_data = []
        for meta, poem in zip(meta_list, poem_list):
            all_data.append(self.single_eval(poem, meta))
        df = pd.DataFrame(all_data)
        df.to_csv(save_file, encoding="utf-8-sig", index=False)
        print(f"评估报表已保存：{save_file}")
        return df

#绘图函数
def draw_compare_fig(rnn_df, trans_df):
    #1. 各题目综合总分对比柱状图
    plt.figure(figsize=(14,6))
    x = np.arange(len(prompt_names))
    width = 0.35
    rnn_total = rnn_df["total_score"].values
    trans_total = trans_df["total_score"].values
    plt.bar(x - width/2, rnn_total, width, label="旧RNN模型", color="#ff7f7f")
    plt.bar(x + width/2, trans_total, width, label="新Transformer模型", color="#63b8ff")
    plt.xticks(x, prompt_names, rotation=30)
    plt.ylabel("综合总分")
    plt.title("7组测试题目综合得分对比")
    plt.legend()
    plt.tight_layout()
    plt.savefig("1_各题目总分对比图.png")
    #2. 五大核心指标全局平均分对比
    indicators = ["format_score", "unique_word_rate", "bigram_repeat", "rhyme_ok", "cangtou_acc"]
    ind_names = ["格式合规分", "词汇丰富度", "短语重复率(越低越好)", "押韵达标率", "藏头准确率"]
    rnn_mean = [
        rnn_df["format_score"].mean(),
        rnn_df["unique_word_rate"].mean(),
        rnn_df["bigram_repeat"].mean(),
        rnn_df["rhyme_ok"].mean(),
        rnn_df["cangtou_acc"].mean()
    ]
    trans_mean = [
        trans_df["format_score"].mean(),
        trans_df["unique_word_rate"].mean(),
        trans_df["bigram_repeat"].mean(),
        trans_df["rhyme_ok"].mean(),
        trans_df["cangtou_acc"].mean()
    ]
    plt.figure(figsize=(12,6))
    x2 = np.arange(len(indicators))
    plt.bar(x2 - width/2, rnn_mean, width, label="RNN", color="#ff7f7f")
    plt.bar(x2 + width/2, trans_mean, width, label="Transformer", color="#63b8ff")
    plt.xticks(x2, ind_names)
    plt.title("全局五大指标平均性能对比")
    plt.legend()
    plt.tight_layout()
    plt.savefig("2_全局指标平均分对比.png")
    #3. 现代题材「都市记忆」专项拆解对比
    rnn_city = rnn_df[rnn_df["prompt"]=="都市记忆"].iloc[0]
    trans_city = trans_df[trans_df["prompt"]=="都市记忆"].iloc[0]
    city_ind = ["format_score","unique_word_rate","bigram_repeat","rhyme_ok"]
    city_v_rnn = [rnn_city[i] for i in city_ind]
    city_v_trans = [trans_city[i] for i in city_ind]
    city_ind_name = ["格式分","词汇丰富度","短语重复率","押韵达标"]
    plt.figure(figsize=(10,5))
    x3 = np.arange(len(city_ind))
    plt.bar(x3-width/2, city_v_rnn, width, label="RNN", color="#ff7f7f")
    plt.bar(x3+width/2, city_v_trans, width, label="Transformer", color="#63b8ff")
    plt.xticks(x3, city_ind_name)
    plt.title("现代题材【都市记忆】分项指标对比（泛化能力测试）")
    plt.legend()
    plt.tight_layout()
    plt.savefig("3_都市记忆专项对比图.png")
    print("三张对比图表已保存到文件夹！")

#读取外部txt诗句文件函数
def load_poem_from_txt(file_path):
    """读取txt，每行一首诗，自动过滤空行"""
    poem_list = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    poem_list.append(line)
    except FileNotFoundError:
        raise Exception(f"错误：找不到{file_path}，请在代码同目录创建该txt并填入7行诗句！")
    #校验必须正好7首，匹配测试集数量
    if len(poem_list) != len(test_set):
        raise Exception(f"文件{file_path}内诗句数量{len(poem_list)}，必须为{len(test_set)}行！")
    return poem_list

#模型生成预留接口
def model_generate(meta_info):
    prompt = meta_info["prompt"]
    mode = meta_info["mode"]
    genre = meta_info["genre"]
    if mode == "prefix_continue":
        input_text = f"以{prompt}为开头，续写一首五言绝句"
    elif mode == "cangtou":
        input_text = f"以{prompt}为题，创作一首七言藏头绝句"
    else:
        word_type = "五言" if genre == "wuyan_jueju" else "七言"
        input_text = f"以{prompt}为题，创作一首{word_type}绝句"
    #替换transformer推理代码
    poem = ""
    return poem

def auto_quality_generate(meta, evaluator):
    final_poem = ""
    for times in range(MAX_RETRY):
        raw_poem = model_generate(meta)
        if not raw_poem.strip():
            print(f"【{meta['prompt']}】第{times+1}次生成空，重试")
            continue
        eval_res = evaluator.single_eval(raw_poem, meta)
        pass_flag, msg = evaluator.is_qualified(eval_res, meta)
        if pass_flag:
            final_poem = raw_poem
            print(f"【{meta['prompt']}】达标：{msg}")
            break
        else:
            print(f"【{meta['prompt']}】不合格：{msg}，自动重生成")
    if final_poem == "":
        final_poem = raw_poem
        print(f"【{meta['prompt']}】重试{MAX_RETRY}次仍未达标，保留最后结果")
    return final_poem

#主程序入口
if __name__ == "__main__":
    eva = PoemEvaluator()
    print("诗词自动评测+外置文件对比工具")
    print("1：模型联动生成（不合格自动重生成，过滤高质量诗句）")
    print("2：批量对比RNN&Transformer【读取外部txt文件】+自动绘图3")
    print("3：单首手动输入测试")
    opt = input("输入数字选择运行模式：")
    if opt == "1":
        print("\n自动生成+质量过滤流水线")
        qualified_poems = []
        for item in test_set:
            good_p = auto_quality_generate(item, eva)
            qualified_poems.append(good_p)
        eva.batch_run(test_set, qualified_poems, "Transformer_过滤后高质量.csv")
    elif opt == "2":
        print("\n批量对比模式：读取2模型生成的诗句")
        rnn_poems = load_poem_from_txt("rnn_poem.txt")
        trans_poems = load_poem_from_txt("trans_poem.txt")
        #生成两份独立CSV评估表
        df_rnn = eva.batch_run(test_set, rnn_poems, "RNN旧模型原始.csv")
        df_trans = eva.batch_run(test_set, trans_poems, "Transformer新模型.csv")
        #自动绘制三张对比图
        draw_compare_fig(df_rnn, df_trans)
    elif opt == "3":
        print("\n单首手动评测")
        for idx, t in enumerate(test_set):
            print(f"{idx+1}. {t['prompt']}")
        num = int(input("输入题目序号：")) - 1
        meta = test_set[num]
        poem = input("粘贴完整诗句：")
        res = eva.single_eval(poem, meta)
        ok, msg = eva.is_qualified(res, meta)
        print("\n评测全部指标：")
        for k,v in res.items():
            print(f"{k}: {v}")
        print(f"是否合格：{ok} | 判定原因：{msg}")