import pandas as pd


def get_mean_length():
    path = "D:\\python_code\\paper\\data\\train4.csv"
    data = pd.read_csv(path)['text']
    lens = [len(s) for s in data]
    mean_len = sum(lens) / len(lens)
    print(mean_len)


if __name__ == '__main__':
    get_mean_length()
