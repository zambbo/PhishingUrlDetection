import pandas as pd
import numpy
import json
def get_json(_fileName):
    try: 
        with open(_fileName, "r") as json_file:
            return json.load(json_file)
    except:
        raise Exception(f"Error while open {_fileName}")    


# label_type = 0 일 경우 benign 1 일 경우 phishing
def toDataFrame(_urlList, label_type:int = 0):
    if label_type == 0:
        labels = numpy.zeros(len(_urlList))
    elif label_type == 1:
        labels = numpy.ones(len(_urlList))
    else:
        raise Exception("Wrong Label Type")
    
    df = pd.DataFrame({
        'url' : _urlList,
        'label' : labels
    })
    return df


def main():
    benign_path = './dataset/after_extract_benign_url.json'
    phishing_path = './dataset/after_extract_phishing_url.json'
    saving_benign_path = './dataset/benigns.csv'
    saving_phishing_path = './dataset/phishings.csv'

    print("start!")
    benign_url_list = get_json(benign_path)
    print("benign json loading finish!")
    phishing_url_list = get_json(phishing_path)
    print("phishing json loading finish!")

    try:
        benign_df = toDataFrame(benign_url_list, 0)
    except Exception as e:
        print(e)
    try:
        phishing_df = toDataFrame(phishing_url_list, 1)
    except Exception as e:
        print(e)
    print("convert to datafrme finish!")

    benign_df.to_csv(saving_benign_path)
    phishing_df.to_csv(saving_phishing_path)
    print("finish all process")

if __name__ == '__main__':
    main()