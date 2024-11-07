import os
import numpy as np
import pandas as pd

if __name__ == "__main__":

    root = "/ssddata/liuyue/github/VisRAG/logs/generator/MiniCPMV2.6"
    file_temp = "eval_g_{}_1.log"
    datasets = [
        "ArxivQA",
        # "ChartQA",
        # "MP-DocVQA",
        # "InfoVQA",
        # "PlotQA",
        # "SlideVQA",
    ]

    for data in datasets:
        path = f"{root}/{file_temp.format(data)}"

        numerical_true, category_true = 0, 0
        numerical_all, category_all = 0, 0
        numerical_sig = False

        answers = []
        correct = []

        with open(f"{path}", "r") as f:
            lines = f.readlines()
            for line in lines:
                if "answer:" in line: ## answer:
                    ans = line.replace("answer:", "")
                    ans = ans.replace("[", "")
                    ans = ans.replace("]", "")
                    ans = ans.replace("%", "")
                    ans_l = ans.split(",")
                    
                    answers.append(ans_l[0].strip())

                elif ", True" in line:
                    correct.append(1)
                elif ", False" in line:
                    correct.append(0)

        answers = np.array(answers)
        correct = np.array(correct)
        assert len(answers) == len(correct)
        print(answers[:5])
        ### correct
        A_ratio = ((answers == "A") * (correct == 1)).sum() / (correct == 1).sum()
        B_ratio = ((answers == "B") * (correct == 1)).sum() / (correct == 1).sum()
        C_ratio = ((answers == "C") * (correct == 1)).sum() / (correct == 1).sum()
        D_ratio = ((answers == "D") * (correct == 1)).sum() / (correct == 1).sum()

        print("True")
        print(f"A:\t{A_ratio*100:.2f}\nB:\t{B_ratio*100:.2f}\nC:\t{C_ratio*100:.2f}\nD:\t{D_ratio*100:.2f}\n")
        print(f"Total:\t{A_ratio+B_ratio+C_ratio+D_ratio}")
        print("="*30)

        ### not correct
        A_ratio = ((answers == "A") * (correct == 0)).sum() / (correct == 0).sum()
        B_ratio = ((answers == "B") * (correct == 0)).sum() / (correct == 0).sum()
        C_ratio = ((answers == "C") * (correct == 0)).sum() / (correct == 0).sum()
        D_ratio = ((answers == "D") * (correct == 0)).sum() / (correct == 0).sum()
        
        print("False")
        print(f"A:\t{A_ratio*100:.2f}\nB:\t{B_ratio*100:.2f}\nC:\t{C_ratio*100:.2f}\nD:\t{D_ratio*100:.2f}\n")
        print(f"Total:\t{A_ratio+B_ratio+C_ratio+D_ratio}")
        print("="*30)
    pass