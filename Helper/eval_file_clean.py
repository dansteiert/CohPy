import os
import pandas as pd
def clean():
    # eval_dir = os.path.join(os.getcwd(), "data", "Evaluation", "christies ’styles’")
    eval_dir = os.path.join(os.getcwd(), "data", "Evaluation", "Lüdtke")
    # search_dir = os.path.join(os.getcwd(), "data", "New_documents")
    for i in os.listdir(eval_dir):
        if i not in ["Planeten.txt", "Pyramide.txt", "Rose1.txt"]:
            df = pd.read_csv(os.path.join(eval_dir, i), dtype=str)
            label_list = ["Mean(VERSTÄNDLICH)", "Mean(Mean_Verstaendlichkeit)", "Mean(Lesbarkeitsrating)"]
            # col_name = "Mean(Mean(WORD_GAZE_DURATION))"
            for j in label_list:
                if j in df.columns:
                    col_name = j
                    break

            list_p = [str(i).replace(",", ".") for i in df[col_name].tolist()]
            df[col_name] = [float(i) for i in list_p]
            # df[col_name] = df.apply(lambda x: float(x[col_name].replace(",", ".")), axis=1)
            df.to_csv(os.path.join(eval_dir, i[:-4] + "_cleanded.csv"))
        
clean()