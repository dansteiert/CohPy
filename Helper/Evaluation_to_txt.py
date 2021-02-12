import os
import pandas as pd



def write_single_txt_files():
    eval_dir = os.path.join(os.getcwd(), "data", "Evaluation", "christies ’styles’")
    search_dir = os.path.join(os.getcwd(), "data", "New_documents")
    
    for i in os.listdir(eval_dir):
        if i not in ["Planeten.txt", "Pyramide.txt", "Rose1.txt"]:
            label_list = ["Mean(VERSTÄNDLICH)", "Mean(Mean_Verstaendlichkeit)", "Mean(Lesbarkeitsrating)"]
            df = pd.read_csv(os.path.join(eval_dir, i))
            # text_col = i[i.find("(") + 1:-5]
            # print(text_col)
            text_col="SENTENCE"
            for index, row in df.iterrows():
                with open(os.path.join(search_dir, i[:-4] + str(index) + ".txt"), "w") as f:
                    f.writelines(str(row[text_col]))
          
          
write_single_txt_files()