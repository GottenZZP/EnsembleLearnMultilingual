import numpy as np
import pandas as pd
from torch import nn
import time
import os
import torch
import random
import json
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import argparse
import warnings

from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import Trainer, TrainingArguments, BertTokenizer, BertModel, BertPreTrainedModel, BertConfig, \
    get_linear_schedule_with_warmup, WEIGHTS_NAME, CONFIG_NAME
from transformers import RobertaTokenizer, AutoTokenizer
from transformers.utils.notebook import format_time
from transformers import logging

from roberta_model import RoBERTaAndTextCnnForSeq, RoBERTa
from distilbert_model import DistilBERTAndTextCnnForSeq, DistilBERT
from albert_modeling import ALBertAndTextCnnForSeq, ALBertForSeq, BigBirdForSeq, LongformerForSeq
from debert_model import Deberta
from xlnet_model import XLNet
from process import InputDataSet, TestInput, my_dataset
from log_set import Log

from d2l import torch as d2l
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logging.set_verbosity_error()
warnings.filterwarnings("ignore")


class HypoParameters:
    def __init__(self):
        self.lr = 3e-5
        self.log_file = '../logging'
        self.data_save_file = '../exp_data'
        self.model_save_file = '../model'
        self.fig_data_save = '../exp_data/'
        self.train_path = '../dataset/data/train_sentiment140.csv'
        self.val_path = '../dataset/data/val_sentiment140.csv'
        self.test_path = '../dataset/data/test_sentiment140.csv'
        self.ans_path = '../exp_data/ans/'
        self.report_path = '../exp_data/report/'
        self.devices = d2l.try_all_gpus()
        # self.devices = [0, 1]
        self.models_config = {
            "roberta_t": '../huggingface/roberta-japanese',
            "roberta": '../huggingface/roberta-japanese',
            "distilbert_t": '../huggingface/distilbert-japanese',
            "distilbert": 'distilbert-base-uncased',
            # "albert": 'ken11/albert-base-japanese-v1-with-japanese-tokenizer',
            'albert': 'albert-base-v2',
            "albert_t": 'ken11/albert-base-japanese-v1-with-japanese-tokenizer',
            "deberta": 'ku-nlp/deberta-v2-base-japanese',
            "xlnet": "hajime9652/xlnet-japanese",
            "bigbird": "google/bigbird-roberta-base",
            "longformer": "allenai/longformer-base-4096"
        }
        self.models_function = {
            "roberta_t": RoBERTaAndTextCnnForSeq,
            "distilbert_t": DistilBERTAndTextCnnForSeq,
            "roberta": RoBERTa,
            "distilbert": DistilBERT,
            "albert_t": ALBertAndTextCnnForSeq,
            'albert': ALBertForSeq,
            'deberta': Deberta,
            'xlnet': XLNet,
            'bigbird': BigBirdForSeq,
            'longformer': LongformerForSeq
        }
        self.filter = [3, 4, 5, 6, 7, 8, 9]
        # self.filter = [2, 3, 4, 5, 6, 7]
        # self.filter = [3, 5, 7, 9, 11]
        # self.filter = [2, 4, 6, 8, 10]
        self.filter_num = 128

    def showParams(self):
        print(f"LR: {self.lr}|filters: {self.filter}|filter num: {self.filter_num}")

    def setHyparams(self, lr,  num_filters, filters):
        self.lr = lr
        self.filter_num = num_filters
        self.filter = filters


# 参数列表
config = HypoParameters()


def set_seed(seed):
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dynamic_graph_plot(loss, epochs):
    fig = plt.figure(figsize=(100, 50), dpi=300)
    plt.xlim(0, epochs)
    plt.xticks(np.arange(epochs))
    plt.xlabel("Epoch")
    plt.ylim(0, 2)
    plt.yticks(np.arange(loss))
    plt.ylabel("Loss")
    plt.title("Validation Loss")
    x, y = [], []

    def graph_up(n):
        x.append(1)
        y.append(n)
        plt.plot(x, y, "r--")

    ani = FuncAnimation(fig, graph_up, )


def cross_valid(train_path, test_path, batch_size, n_splits, tokenizer):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=6)
    train_data, valid_data = [], []
    for train_idx, valid_idx in kf.split(train):
        train_temp = train.iloc[train_idx]
        valid_temp = train.iloc[valid_idx]
        train_temp.index = [x for x in range(len(train_temp))]
        valid_temp.index = [x for x in range(len(valid_temp))]
        train_data.append(train_temp)
        valid_data.append(valid_temp)
    train_iter_list = []
    for data in train_data:
        train_temp = InputDataSet(data, tokenizer, 128)
        train_iter = DataLoader(train_temp, batch_size=batch_size, num_workers=0)
        train_iter_list.append(train_iter)
    valid_iter_list = []
    for data in valid_data:
        valid_temp = InputDataSet(data, tokenizer, 128)
        valid_iter = DataLoader(valid_temp, batch_size=batch_size, num_workers=0)
        valid_iter_list.append(valid_iter)
    test_data = TestInput(test, tokenizer, 128)
    test_iter = DataLoader(test_data, batch_size=batch_size, num_workers=0)
    return train_iter_list, valid_iter_list, test_iter


def glove_w_e():
    glove_vectors = {}
    with open("D:\python_code\paper_extend\dataset\data\Glove_IMDB_768d.txt", 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            glove_vectors[word] = vector
    return glove_vectors


def train(batch_size, epochs, model_type, is_save=True, is_load_glove=False):
    # choose tokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased') if model_type in ["distilbert", "distilbert_t"] else \
        AutoTokenizer.from_pretrained(config.models_config[model_type])
    # 加载预训练
    model = config.models_function[model_type].from_pretrained(config.models_config[model_type], config.filter, config.filter_num) if len(model_type.split('_')) > 1 else \
        config.models_function[model_type].from_pretrained(config.models_config[model_type])

    train_data, val_data = my_dataset(config.train_path, tokenizer, test_path=config.val_path)

    if is_load_glove:
        # model.albert.embeddings.word_embeddings.load_state_dict(torch.load(args.glove_path))
        glove_vectors = glove_w_e()
        # 获取ALBERT模型的词向量矩阵
        albert_embeddings = model.model.embeddings.word_embeddings.weight.data.numpy()

        # 将Glove词向量映射到ALBERT模型的词汇表中
        for word, index in tokenizer.vocab.items():
            if word in glove_vectors:
                albert_embeddings[index] = glove_vectors[word]

        # 更新BERT模型的词向量矩阵
        model.model.embeddings.word_embeddings.weight.data.copy_(torch.from_numpy(albert_embeddings))
        print("加载Glove成功！")

    train_iter = DataLoader(train_data, batch_size=batch_size, num_workers=16)
    val_iter = DataLoader(val_data, batch_size=batch_size, num_workers=16)

    optimizer = AdamW(model.parameters(), lr=config.lr)
    # optimizer = nn.DataParallel(optimizer, device_ids=config.devices)
    print(len(train_data))
    print(len(train_iter))
    total_steps = len(train_iter) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.05 * total_steps,
        num_training_steps=total_steps)

    total_time_start = time.time()

    file_lens = len(os.listdir(os.path.join(config.model_save_file, model_type)))
    info_name = f"{time.strftime('%Y-%m-%d-%H-%M')}"

    logger = Log(f'{file_lens}-model-{info_name}.log', model_type)

    logger.writer_log(f"Train batch size: {batch_size}")
    logger.writer_log(f"Learning rate: {config.lr}")
    logger.writer_log(f"Total steps: {total_steps}")
    logger.writer_log(f"Filter type: {config.filter}")
    logger.writer_log(f"Filter number: {config.filter_num}")
    logger.writer_log(f"Training Start!")

    # 最大的一次epoch准确率
    max_val_f1 = 0.0
    # 准确率没上升次数
    repeat_acc = 0

    # 每个batch的loss值
    total_loss = []
    plot_loss = []
    plot_acc = []
    total_step = 0

    # 多GPU跑
    model = nn.DataParallel(model, device_ids=config.devices)
    # model = model.cuda()
    model = model.to(config.devices[0])

    for epoch in range(epochs):
        total_train_loss = 0
        t0 = time.time()
        # train_iter.sampler.set_epoch(epoch)

        # model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        model.train()
        bar = tqdm(enumerate(train_iter), total=len(train_iter))
        for step, batch in bar:
            # 解包
            input_ids = batch["input_ids"].to(config.devices[0])
            attention_mask = batch["attention_mask"].to(config.devices[0])
            labels = batch["labels"].to(config.devices[0])

            model.zero_grad()
            # 获取输出
            outputs = model(input_ids, attention_mask, labels)
            logits = torch.argmax(outputs.logits, dim=1)
            # 将预测值不参与后续训练集的梯度计算
            preds = logits.detach().cpu().numpy()
            labels_ids = labels.to("cpu").numpy()
            # 求出该批次的准确率
            acc = (preds == labels_ids).mean()
            plot_acc.append(acc)
            loss = outputs.loss
            # loss = loss.item()
            loss = loss.mean()
            # 将每轮的损失累加起来
            total_train_loss += loss
            plot_loss.append(loss.item())
            if step % 100 == 0:
                total_loss.append(loss.item())
            total_step += 1
            # loss.mean().backward()
            loss.backward()
            # 梯度裁剪（防止梯度爆炸）
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            bar.set_description(f'Epoch [{epoch + 1}/{epochs}]')
            bar.set_postfix(loss=loss.item(), acc=acc)

        avg_train_loss = total_train_loss / len(train_iter)

        now_lr = optimizer.param_groups[0]["lr"]

        train_time = format_time(time.time() - t0)
        val_time_start = time.time()

        logger.writer_log('')
        logger.writer_log(f"====Epoch:[{epoch + 1}/{epochs}] || avg_train_loss={avg_train_loss:.3f} || LR={now_lr}====")
        logger.writer_log(f"====Training epoch took: {train_time}====")
        logger.writer_log("Running Validation...")

        model.eval()
        # 获取验证集的误差和准确度
        avg_val_loss, avg_val_acc, val_pre, val_rec, val_f1, val_pred = evaluate(model, val_iter)

        val_time = format_time(time.time() - val_time_start)

        logger.writer_log(f"====Epoch:[{epoch + 1}/{epochs}] || avg_val_loss={avg_val_loss:.3f} || avg_val_acc={avg_val_acc:.4f}==== || val_pre,recall,f1={val_pre, val_rec, val_f1}====")
        logger.writer_log(f"====Validation epoch took: {val_time}====")
        logger.writer_log('')

        if val_f1 <= max_val_f1:
            repeat_acc += 1

        # 若准确率比最大的epoch更好时将模型保存起来
        elif val_f1 > max_val_f1:
            max_val_f1 = val_f1
            repeat_acc = 0

            if is_save:
                from prediction import save_file
                save_file(val_pred, model_type + str(file_lens))

                output_dir = config.model_save_file
                output_name = f"{file_lens}-model.bin"
                output_model_file = os.path.join(os.path.join(output_dir, model_type), output_name)
                if isinstance(model, torch.nn.DataParallel):
                    torch.save(model.module.state_dict(), output_model_file)
                else:
                    torch.save(model.state_dict(), output_model_file)
                # print("Model saved!")

    logger.writer_log('', "DEBUG")
    logger.writer_log('Training Completed!', "DEBUG")
    logger.writer_log(f'Max F1-score is {max_val_f1}')
    print(f"Total train time: {format_time(time.time() - total_time_start)}")

    model_acc_list = os.listdir('../exp_data/acc')
    model_loss_list = os.listdir('../exp_data/loss')
    model_acc_len = model_loss_len = 0
    for model_name in model_acc_list:
        if model_name[:len(model_type)] == model_type:
            model_acc_len += 1
    for model_name in model_loss_list:
        if model_name[:len(model_type)] == model_type:
            model_loss_len += 1
    df_loss, df_acc = pd.DataFrame(plot_loss, columns=[model_type]), pd.DataFrame(plot_acc, columns=[model_type])
    df_loss.to_csv(f"../exp_data/loss/{model_type}-{model_loss_len}.csv", header=False)
    df_acc.to_csv(f"../exp_data/acc/{model_type}-{model_acc_len}.csv", header=False)


def k_fold_train(batch_size, epochs, k_fold, model_type):
    from prediction import my_prediction, avg_prediction

    tokenizer = AutoTokenizer.from_pretrained('../huggingface/roberta-japanese') if model_type in ["distilbert", "distilbert_t"] else \
        AutoTokenizer.from_pretrained(config.models_config[model_type])

    train_iter_list, valid_iter_list, test_iter = cross_valid(config.train_path, config.val_path, batch_size, k_fold, tokenizer)

    total_time_start = time.time()

    file_lens = len(os.listdir(os.path.join(config.model_save_file, model_type)))
    info_name = f"{time.strftime('%Y-%m-%d-%H-%M')}"
    logger = Log(f'{file_lens}-model-{info_name}.log', model_type)

    logger.writer_log(f"Train batch size: {batch_size}")
    logger.writer_log(f"Learning rate: {config.lr}")
    logger.writer_log(f"Filter type: {config.filter}")
    logger.writer_log(f"Filter number: {config.filter_num}")
    logger.writer_log(f"Training Start!")

    total_max_f1 = 0.0
    total_max_pre = 0.0
    total_max_rec = 0.0
    total_max_acc = 0.0

    final_acc = []
    final_loss = []

    k_result = []
    true_label = []

    model = config.models_function[model_type].from_pretrained(config.models_config[model_type], config.filter,
                                                               config.filter_num) if len(model_type.split('_')) > 1 else \
        config.models_function[model_type].from_pretrained(config.models_config[model_type])
    optimizer = AdamW(model.parameters(), lr=config.lr)
    # 多GPU跑
    model = nn.DataParallel(model, device_ids=config.devices)
    model = model.to(config.devices[0])

    for k, (train_iter, valid_iter) in enumerate(zip(train_iter_list, valid_iter_list)):
        plot_loss = []
        plot_acc = []

        total_steps = len(train_iter) * epochs

        logger.writer_log(f"Total steps: {total_steps}")

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0.1 * total_steps,
            num_training_steps=total_steps)

        for epoch in range(epochs):
            total_train_loss = 0
            t0 = time.time()
            # train_iter.sampler.set_epoch(epoch)
            model.train()
            bar = tqdm(enumerate(train_iter), total=len(train_iter))
            for step, batch in bar:
                # 解包
                input_ids = batch["input_ids"].to(config.devices[0])
                attention_mask = batch["attention_mask"].to(config.devices[0])
                labels = batch["labels"].to(config.devices[0])

                model.zero_grad()
                # 获取输出
                outputs = model(input_ids, attention_mask, labels)
                logits = torch.argmax(outputs.logits, dim=1)
                # 将预测值不参与后续训练集的梯度计算
                preds = logits.detach().cpu().numpy()
                labels_ids = labels.to("cpu").numpy()
                # 求出该批次的准确率
                acc = (preds == labels_ids).mean()
                plot_acc.append(acc)
                loss = outputs.loss
                loss = loss.mean()
                # 将每轮的损失累加起来
                total_train_loss += loss.item()
                plot_loss.append(loss.item())

                loss.backward()
                # 梯度裁剪（防止梯度爆炸）
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                bar.set_description(f'K fold [{k + 1}/{k_fold}], Epoch [{epoch + 1}/{epochs}]')
                bar.set_postfix(loss=loss.item(), acc=acc)

            avg_train_loss = total_train_loss / len(train_iter)

            now_lr = optimizer.param_groups[0]["lr"]

            train_time = format_time(time.time() - t0)
            val_time_start = time.time()

            logger.writer_log('')
            logger.writer_log(f"====Epoch:[{epoch + 1}/{epochs}] || avg_train_loss={avg_train_loss:.3f} || LR={now_lr}====")
            logger.writer_log(f"====Training epoch took: {train_time}==== || k is {k + 1}")
            logger.writer_log("Running Validation...")

            model.eval()
            # 获取验证集的误差和准确度
            avg_val_loss, avg_val_acc, val_pre, val_rec, val_f1, val_pred = evaluate(model, valid_iter)

            val_time = format_time(time.time() - val_time_start)

            logger.writer_log(
                f"====Epoch:[{epoch + 1}/{epochs}] || avg_val_loss={avg_val_loss:.3f} || avg_val_acc={avg_val_acc:.3f}==== || val_pre,recall,f1={val_pre, val_rec, val_f1}====")
            logger.writer_log(f"====Validation epoch took: {val_time}==== || k is {k + 1}")
            logger.writer_log('')

            # 若准确率比最大的epoch更好时将模型保存起来
            if val_f1 > total_max_f1:
                total_max_f1 = val_f1
                total_max_pre = val_pre
                total_max_rec = val_rec
                total_max_acc = avg_val_acc

                final_acc = plot_acc
                final_loss = plot_loss

                from prediction import save_file
                save_file(val_pred, model_type + str(file_lens))

                output_dir = config.model_save_file
                output_name = f"{file_lens}-model.bin"
                output_model_file = os.path.join(os.path.join(output_dir, model_type), output_name)
                if isinstance(model, torch.nn.DataParallel):
                    torch.save(model.module.state_dict(), output_model_file)
                else:
                    torch.save(model.state_dict(), output_model_file)

        # lst_prob, lst_true = my_prediction(model, test_iter, "D:\python_code\paper\data\\test_label2.csv",
        #                                    info_name, config.devices[0])

        lst_prob, lst_true = my_prediction(model, test_iter, "D:\python_code\paper_extend\dataset\data\\val3_label.csv",
                                           info_name, config.devices[0])
        k_result.append(lst_prob)
        true_label = lst_true

        logger.writer_log('', "DEBUG")
        logger.writer_log(f'The {k + 1} fold was training Completed!', "DEBUG")
        logger.writer_log(f'Max Precision and Recall and F1-score is {total_max_pre, total_max_rec, total_max_f1}')

    acc, p_s, r_s, f1_macro, avg_preds = avg_prediction(k_result, true_label)
    avg_preds = pd.DataFrame(avg_preds)
    avg_preds.to_csv(f"../exp_data/ans/{model_type}_valid_ensemble.csv")

    logger.writer_log(f"Test dataset avg acc: {acc}, precision: {p_s}, recall: {r_s}, f1_macro: {f1_macro}")
    logger.writer_log(f"Valid dataset max acc: {total_max_acc}, precision: {total_max_pre}, recall: {total_max_rec}, f1_macro: {total_max_f1}")
    print(f"Total train time: {format_time(time.time() - total_time_start)}")

    df_loss, df_acc = pd.DataFrame(final_loss, columns=[model_type]), pd.DataFrame(final_acc, columns=[model_type])
    df_loss.to_csv(f"../exp_data/loss/{model_type}-ensemble.csv", header=False)
    df_acc.to_csv(f"../exp_data/acc/{model_type}-ensemble.csv", header=False)


def evaluate(model, val_iter):
    """计算验证集的误差和准确率"""
    total_val_loss = 0
    corrects = []
    total_label_true = []
    total_label_pred = []
    for batch in val_iter:
        # 从迭代器中取出每个批次
        input_ids = batch["input_ids"].to(config.devices[0])
        attention_mask = batch["attention_mask"].to(config.devices[0])
        labels = batch["labels"].to(config.devices[0])

        # 验证集的outputs不参与训练集后续的梯度计算
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, labels)

        # 获取该批次中所有样本的所有分类中的最大值，将最大值变为1，其余变为0
        logits = torch.argmax(outputs.logits, dim=1)
        # 将预测值不参与后续训练集的梯度计算
        preds = logits.detach().cpu().numpy()
        labels_ids = labels.to("cpu").numpy()
        # 求出该批次的准确率
        corrects.append((preds == labels_ids).mean())

        total_label_pred.extend(list(preds))
        total_label_true.extend(list(labels_ids))

        loss = outputs.loss
        loss = loss.mean()
        # 累加损失
        total_val_loss += loss.item()

    # 求出平均损失
    avg_val_loss = total_val_loss / len(val_iter)
    # 求出平均准确率
    # avg_val_acc = np.mean(corrects)

    val_acc = accuracy_score(total_label_true, total_label_pred)
    val_f1 = f1_score(total_label_true, total_label_pred, average='macro')
    val_pre = precision_score(total_label_true, total_label_pred, average='macro')
    val_rec = recall_score(total_label_true, total_label_pred, average='macro')

    return avg_val_loss, val_acc, val_pre, val_rec, val_f1, total_label_pred


def main():
    set_seed(2023)
    # config.setHyparams(3e-5, [2, 3, 4, 5, 6, 7], 128)
    train(32, 5, 'albert')
    # k_fold_train(320, 20, 10, 'xlnet')


if __name__ == '__main__':
    main()
