import os
import numpy as np
import pandas as pd

if __name__ == "__main__":

    root = "/ssddata/liuyue/github/VisRAG/logs/generate/LLaVA-ov-0.5b"
    file_temp = "eval_g_{}_1.log"
    datasets = [
        "ArxivQA",
        "ChartQA",
        "MP-DocVQA",
        "InfoVQA",
        "PlotQA",
        "SlideVQA",
    ]

    for data in datasets:
        path = f"{root}/{file_temp.format(data)}"

        numerical_true, category_true = 0, 0
        numerical_all, category_all = 0, 0
        numerical_sig = False

        with open(f"{path}", "r") as f:
            lines = f.readlines()
            for line in lines:
                if "answer:" in line: ## answer:
                    ans = line.replace("answer:", "")
                    ans = ans.replace("[", "")
                    ans = ans.replace("]", "")
                    ans = ans.replace("%", "")
                    ans_l = ans.split(",")
                    
                    try:
                        float(ans_l[0])
                    except:
                        category_all += 1
                    else:
                        numerical_all += 1
                        numerical_sig = True

                elif ", True" in line:
                    if numerical_sig:
                        numerical_true += 1
                        numerical_sig = False
                    else:
                        category_true += 1
                    
        df = pd.DataFrame({
            "correct": [numerical_true, category_true],
            "all": [numerical_all, category_all],
            "correct_rate": [numerical_true/(numerical_all+1e-7), category_true/(category_all+1e-7)]
        })
        df.index = ["numerical", "category"]

        print(f"=== {data} ===")
        print(df)
        print(f"==============")
        print()
    pass