import os
import pandas as pd



def write_single_txt_files():
    eval_dir = os.path.join("D:\\Uni\\NLP\\Projects\\CohPy", "data", "Evaluation", "christies ’styles’")
    # eval_dir = os.path.join("D:\\Uni\\NLP\\Projects\\CohPy", "data", "Evaluation", "Lüdtke")
    search_dir = os.path.join("D:\\Uni\\NLP\\Projects\\CohPy", "data", "New_documents", "en")
    # search_dir = os.path.join("D:\\Uni\\NLP\\Projects\\CohPy", "data", "New_documents", "de")
    
    for i in os.listdir(eval_dir):
        if "_cleanded.csv" in i:
            df = pd.read_csv(os.path.join(eval_dir, i))
            
            # # Christie
            text_col="SENTENCE"
            text_list = df[text_col].tolist()
            text = " ".join(text_list)
            with open(os.path.join(search_dir, i[:-4] + ".txt"), "w") as f:
                f.write(text)
            for index, row in df.iterrows():
                with open(os.path.join(search_dir, i[:-13] + str(index) + ".txt"), "w") as f:
                    f.writelines(str(row[text_col]))
            ## Lüdtke
            # text_col = i[i.find("(") + 1:-14]
            # for index, row in df.iterrows():
            #     with open(os.path.join(search_dir, i[:-13] + str(index) + ".txt"), "w") as f:
            #         f.writelines(str(row[text_col]))
                    
        ## Lüdtke
        elif i in ["Planeten.txt", "Pyramide.txt", "Rose1.txt"]:
            with open(os.path.join(eval_dir, i), "r") as read:
                text_list = read.readlines()
                with open(os.path.join(search_dir, i), "w") as file:
                    file.writelines(text_list)
          
          
write_single_txt_files()