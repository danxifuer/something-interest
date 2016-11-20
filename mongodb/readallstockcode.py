import pandas as pd

def readallcode():
    filePath = "/home/daiab/code/ml/something-interest/mongodb/allstockcode.csv"
    csv = pd.read_csv(filepath_or_buffer=filePath, index_col=0, dtype=str)
    return fiterCode(csv['code'].values)

def fiterCode(codeList):
    filter_result = []
    for code in codeList:
        if code.startswith("3"):continue
        filter_result.append(code)
    return filter_result


if __name__ == '__main__':
    print(readallcode())