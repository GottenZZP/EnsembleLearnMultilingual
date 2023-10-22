import numpy
import numpy as np
import pandas as pd
from torch import nn
import time
import os
import torch
import logging
from torch.optim import AdamW
from transformers import Trainer, TrainingArguments, BertTokenizer, BertModel, BertPreTrainedModel, BertConfig, \
    get_linear_schedule_with_warmup, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
# from transformers.utils.notebook import format_time
from roberta_model import RoBERTa, RoBERTaAndTextCnnForSeq
from distilbert_model import DistilBERT, DistilBERTAndTextCnnForSeq
from process import InputDataSet, TestInput, read_file, process_text
from d2l import torch as d2l
from sklearn.metrics import classification_report
from train import HypoParameters


device = d2l.try_gpu()
config = HypoParameters()


def format_time(t):
    """Format `t` (in seconds) to (h):mm:ss"""
    t = int(t)
    h, m, s = t // 3600, (t // 60) % 60, t % 60
    return f"{h}:{m:02d}:{s:02d}" if h != 0 else f"{m:02d}:{s:02d}"


def get_result(pred, lst_true):
    """Get final result"""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    acc = accuracy_score(lst_true, pred)
    p_s = precision_score(lst_true, pred, average='macro')
    r_s = recall_score(lst_true, pred, average='macro')
    f1_macro = f1_score(lst_true, pred, average='macro')

    return acc, p_s, r_s, f1_macro


def avg_prediction(k_result, lst_true):
    k_result = np.array(k_result)
    avg_probs = np.sum(k_result, axis=0) / 5
    avg_probs = torch.from_numpy(avg_probs)
    avg_preds = torch.argmax(avg_probs, dim=1)
    acc, p_s, r_s, f1_macro = get_result(avg_preds, lst_true)
    return acc, p_s, r_s, f1_macro, avg_preds


def temp(model):
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float).to(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings = np.zeros((repetitions, 1))
    # GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    mean_fps = 1000. / mean_syn
    print(' * Mean@1 {mean_syn:.3f}ms Std@5 {std_syn:.3f}ms FPS@1 {mean_fps:.2f}'.format(mean_syn=mean_syn,
                                                                                         std_syn=std_syn,
                                                                                         mean_fps=mean_fps))
    print(mean_syn)


def model_prediction(test_iter, model, model_type):
    # model.load_state_dict(torch.load("D:\\python_code\\金融评论分类\\cache\\model_stu.bin"))
    checkpoint = torch.load(f"../model/{model_type}/2-model.bin")
    model.load_state_dict(checkpoint)
    model = model.to(device)
    corrects = []
    model.eval()
    print("Evaluate Start!")
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((len(test_iter), 1))
    start_time = time.time()
    for step, batch in enumerate(test_iter):
        print(f"The [{step + 1}]/[{len(test_iter)}]")
        with torch.no_grad():
            starter.record()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            # token_type_ids = batch["token_type_ids"].to(device)

            outputs = model(input_ids, attention_mask)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[step] = curr_time

            logits = torch.argmax(outputs.logits, dim=1)
            preds = logits.detach().cpu().numpy()

            corrects.append(preds)
    print(f"average time is {format_time((time.time() - start_time) / 16)}")
    mean_syn = np.sum(timings) / (len(test_iter) * 128)
    std_syn = np.std(timings)
    mean_fps = 1000. / mean_syn
    print(' * Mean@1 {mean_syn:.3f}ms Std@5 {std_syn:.3f}ms FPS@1 {mean_fps:.2f}'.format(mean_syn=mean_syn,
                                                                                         std_syn=std_syn,
                                                                                         mean_fps=mean_fps))
    print(mean_syn)
    print("Evaluate End!")
    return corrects


def save_file(corrects, model_type):
    total_ans = []
    # if isinstance(corrects, list):
    #     for batch in corrects:
    #         for ans in batch:
    #             total_ans.append(int(ans))
    # else:
    total_ans = corrects

    df1 = pd.read_csv("D:\python_code\paper\data\idx.csv")
    idx, labels = df1["idx"], df1["label"]

    index_to_label = dict()
    for l, i in zip(labels, idx):
        index_to_label[i] = l

    final_ans = []
    for n in total_ans:
        final_ans.append([n, index_to_label[n]])

    df = pd.DataFrame(final_ans, columns=["idx", "label"])
    out_path = config.ans_path + model_type + '_ans.csv'
    df.to_csv(out_path, index=True, sep=',')
    return out_path


def acc_rate(label_path, ans_path):
    df1 = pd.read_csv(label_path)
    df2 = pd.read_csv(ans_path)

    test_label = df1["idx"]
    ans_label = df2["idx"]
    acc = (test_label == ans_label).mean()
    for i, ta in enumerate(zip(test_label, ans_label)):
        t, a = ta
        if t != a:
            print(i)
    return acc


def getEvaReport(test_label, test_pred, file_name):
    df1 = pd.read_csv(test_label)
    df2 = pd.read_csv(test_pred)
    labels = df1["idx"].to_list()
    preds = df2["idx"].to_list()
    target_names = pd.read_csv("D:\python_code\paper\data\idx.csv")["label"].to_list()
    # target_names = [f"class: {x}" for x in target_names]
    res = classification_report(y_true=labels, y_pred=preds, target_names=target_names, digits=4, output_dict=True)
    # print(res)
    df3 = pd.DataFrame(res)
    out = pd.DataFrame(df3.values.T, index=df3.columns, columns=df3.index)
    print(out)
    out_path = config.report_path + file_name + '.csv'
    out.to_csv(out_path)


def Ensemble_learning(albert, roberta, distilbert):
    df1 = pd.read_csv(albert)['idx'].to_list()
    df2 = pd.read_csv(roberta)['idx'].to_list()
    df3 = pd.read_csv(distilbert)['idx'].to_list()

    final_res = []
    for a, r, d in zip(df1, df2, df3):
        if a == r:
            final_res.append(a)
        elif a == d:
            final_res.append(a)
        elif r == d:
            final_res.append(r)
        elif a != r != d:
            final_res.append(r)
    df = pd.DataFrame(final_res, columns=["idx"])
    df.to_csv("../exp_data/final_label.csv")


def my_prediction(model, testing_loader, test_label, info_name, device):
    """Prediction function"""

    final_file = os.path.join("D:\python_code\paper\data\preds", info_name + "-preds.txt")
    labels = pd.read_csv(test_label)["idx"]
    labels = np.array([x for x in labels])
    lst_prediction = []
    lst_true = []
    lst_prob = []
    model.eval()
    print("Evaluate Start!")
    for step, batch in enumerate(testing_loader):
        print(f"The [{step + 1}]/[{len(testing_loader)}]")
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            logits = torch.argmax(probs, dim=1)
            preds = logits.detach().cpu().numpy()

            lst_prediction.append(preds)
            lst_prob.append(probs)
    print("Evaluate End!")

    lst_true = [int(l) for l in labels]
    lst_prediction = [int(i) for l in lst_prediction for i in l]
    lst_prob = [i.to('cpu').numpy() for prob in lst_prob for i in prob]

    return lst_prob, lst_true


if __name__ == "__main__":
    model_type = "longformer"
    test = read_file(config.test_path)
    test = process_text(test)
    tokenizer = AutoTokenizer.from_pretrained(config.models_config[model_type])
    # tokenizer = AutoTokenizer.from_pretrained('albert-base-v2')
    test_data = TestInput(test, tokenizer, 512)
    test_iter = DataLoader(test_data, batch_size=64)
    print(len(test_data))
    # # model = config.models_function[model_type].from_pretrained(config.models_config[model_type])
    model = config.models_function[model_type].from_pretrained(config.models_config[model_type], config.filter,
                                                               config.filter_num) if len(model_type.split('_')) > 1 else \
        config.models_function[model_type].from_pretrained(config.models_config[model_type])
    # corrects = model_prediction(test_iter, model)
    # preds_path = save_file(corrects, model_type)
    # # getEvaReport("D:\python_code\paper\data\\test_label2.csv", preds_path, model_type)
    # getEvaReport("D:\python_code\paper_extend\dataset\data\\val3_label.csv", preds_path, model_type)

    # Ensemble_learning("D:\python_code\paper_extend\exp_data\\ans\\albert_t_valid_ensemble4.csv", "D:\python_code\paper_extend\exp_data\\ans\\roberta_t_valid_ensemble.csv", "D:\python_code\paper_extend\exp_data\\ans\distilbert_t_valid_ensemble.csv")
    # getEvaReport("D:\python_code\paper_extend\dataset\data\\val3_label.csv", "D:\python_code\paper_extend\exp_data\\final_label.csv", "Ensemble2")

    model_prediction(test_iter, model, model_type)
