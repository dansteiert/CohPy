import os
import pandas as pd



def write_single_txt_files():
    # eval_dir = os.path.join("D:\\Uni\\NLP\\Projects\\CohPy", "data", "Evaluation", "christies ’styles’")
    # search_dir = os.path.join("D:\\Uni\\NLP\\Projects\\CohPy", "data", "New_documents", "en")

    eval_dir = os.path.join("D:\\Uni\\NLP\\Projects\\CohPy", "data", "Evaluation", "Lüdtke")
    search_dir = os.path.join("D:\\Uni\\NLP\\Projects\\CohPy", "data", "New_documents", "de")
    
    
    file_list = []
    label_list = []
    for i in os.listdir(eval_dir):
        if "_cleanded.csv" in i:
            df = pd.read_csv(os.path.join(eval_dir, i))

            # # # Christie
            # text_col = "SENTENCE"
            # label_col = "Mean(Mean(WORD_GAZE_DURATION))"
            # text_list = df[text_col].tolist()
            # text = " ".join(text_list)
            # with open(os.path.join(search_dir, i[:-4] + ".txt"), "w") as f:
            #     f.write(text)
            #
            # for index, row in df.iterrows():
            #     try:
            #         with open(os.path.join(search_dir, i[:-13] + str(index) + ".txt"), "w") as f:
            #             f.writelines(str(row[text_col]) + " " + str(df.iloc[index + 1][text_col]))
            #         file_list.append(i[:-13] + str(index) + ".txt")
            #         label_list.append((row[label_col] + df.iloc[index + 1][label_col])/2)
            #     except:
            #         pass

            
            
            ## Lüdtke
            text_col = i[i.find("(") + 1:-14]
            if i == "Text_Rating_Expra_Nov_2017 By (Text)_cleanded.csv":
                label_col = "Mean(Mean_Verstaendlichkeit)"
                for index, row in df.iterrows():
                    file_list.append(row[text_col] + ".txt")
                    label_list.append(row[label_col])
            else:
                if i == "Zweisatz_Vorstudie_Verstehen By (Satz1_2)_cleanded.csv":
                    label_col = "Mean(Lesbarkeitsrating)"
                elif i == "SWRF-V_Rohdaten By (ITEM)_cleanded.csv":
                    label_col = "Mean(VERSTÄNDLICH)"
                for index, row in df.iterrows():
                    with open(os.path.join(search_dir, i[:-13] + str(index) + ".txt"), "w") as f:
                        f.writelines(str(row[text_col]))
                    file_list.append(i[:-13] + str(index) + ".txt")
                    label_list.append(row[label_col])
                    
        ## Lüdtke
        elif i in ["Planeten.txt", "Pyramide.txt", "Rose1.txt"]:
            with open(os.path.join(eval_dir, i), "r") as read:
                text_list = read.readlines()
                with open(os.path.join(search_dir, i), "w") as file:
                    file.writelines(text_list)
        result_df = pd.DataFrame(data={"identifier": file_list, "continuous_label": label_list})
        if os.path.isfile(os.path.join("D:\\Uni\\NLP\\Projects\\CohPy", "data", "Evaluation", "Evaluation_label.csv")):
            result_df_temp = pd.read_csv(
                os.path.join("D:\\Uni\\NLP\\Projects\\CohPy", "data", "Evaluation", "Evaluation_label.csv"), usecols=["identifier", "continuous_label"])
            result_df = result_df_temp.append(result_df)
        result_df.to_csv(os.path.join("D:\\Uni\\NLP\\Projects\\CohPy", "data", "Evaluation", "Evaluation_label.csv"))


write_single_txt_files()